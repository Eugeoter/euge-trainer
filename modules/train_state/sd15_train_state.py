import os
import random
from diffusers import DiffusionPipeline
from waifuset import logging
from .base_train_state import BaseTrainState
from ..utils import class_utils, eval_utils, sd15_model_utils

logger = logging.get_logger("train")


class SD15TrainState(BaseTrainState):
    use_mem_eff_save: bool = False
    use_xformers: bool = False

    pipeline_class: type
    eval_benchmark: str = None
    eval_sampler: str = 'euler_a'
    eval_train_size: int = 4
    eval_valid_size: int = 4
    eval_params = class_utils.cfg(
        prompt="1girl, solo, cowboy shot, white background, smile, looking at viewer, serafuku, pleated skirt",
        negative_prompt="abstract, bad anatomy, clumsy pose, signature",
        steps=28,
        batch_size=1,
        batch_count=4,
        scale=7.5,
        seed=42,
        width=832,
        height=1216,
        save_latents=False,
    )

    sample_to_wandb: bool = True

    # @classmethod
    # def from_config(
    #     cls,
    #     config,
    #     trainer,
    #     accelerator,
    #     train_dataset,
    #     valid_dataset,
    #     pipeline_class,
    #     optimizer,
    #     lr_scheduler,
    #     train_dataloader,
    #     valid_dataloader,
    #     save_dtype,
    #     **kwargs,
    # ):
    #     return super().from_config(
    #         config,
    #         trainer=trainer,
    #         accelerator=accelerator,
    #         train_dataset=train_dataset,
    #         valid_dataset=valid_dataset,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #         train_dataloader=train_dataloader,
    #         valid_dataloader=valid_dataloader,
    #         save_dtype=save_dtype,
    #         pipeline_class=pipeline_class,
    #         **kwargs,
    #     )

    def save_diffusion_model(self) -> str:
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        sd15_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.accelerator.unwrap_model(self.nnet),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_path=None,
            vae=self.vae,
            save_dtype=self.save_dtype,
            metadata=None,
            v2=self.v2,
            use_mem_eff_save=self.use_mem_eff_save,
        )
        self.logger.info(f"Diffusion model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def save_train_state_to_disk(self) -> str:
        self.logger.print(f"Saving train state at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_train_state_dir, f"{self.output_name['train_state']}_train-state_ep{self.epoch}_step{self.global_step}")
        with self.logger.timer(f"Save train state to {save_path}"):
            self.accelerator.save_state(save_path)
        return save_path

    def get_sampler(self):
        sampler_kwargs = {}
        if self.prediction_type == 'v_prediction':
            sampler_kwargs['prediction_type'] = 'v_prediction'
        if self.zero_terminal_snr:
            sampler_kwargs['rescale_betas_zero_snr'] = True
        return eval_utils.get_sampler(
            self.eval_sampler,
            **sampler_kwargs,
        )

    def get_pipeline(self) -> DiffusionPipeline:
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.vae),
            scheduler=self.get_sampler(),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )

    def do_eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False, on_train_start=False):
        benchmark = self.get_benchmark()
        if not benchmark:
            self.logger.info("No benchmark found, skipping sample")
            return
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            try:
                sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"ep{self.epoch}_step{self.global_step}")
                pipeline = self.get_pipeline()
                # pipeline.set_use_memory_efficient_attention_xformers(self.use_xformers)
                self.sample_images(pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                self.logger.info(f"Sampled images saved to: `{logging.yellow(sample_dir)}`")
            except Exception as e:
                import traceback
                self.logger.error(logging.red("Exception when sample images:", e))
                traceback.print_exc()
                pass
        self.accelerator.wait_for_everyone()

    def sample_images(
        self,
        pipeline,
        sample_dir,
        benchmark,
        on_epoch_end=False
    ):
        eval_utils.sample_during_train(
            pipeline=pipeline,
            sample_dir=sample_dir,
            benchmark=benchmark,
            default_params=self.eval_params,
            accelerator=self.accelerator,
            epoch=self.epoch if on_epoch_end else None,
            steps=self.global_step,
            device=self.accelerator.device,
            wandb_run=self.wandb_run if self.sample_to_wandb else None,
        )

    def get_sample_seed(self):
        return 42

    def get_benchmark(self):
        is_controlnet = 'controlnet' in self.train_dataset.__class__.__name__.lower()
        if self.eval_benchmark is not None:
            return self.eval_benchmark
        else:
            ind_lst = []
            for i in range(self.eval_train_size):
                random.seed(self.get_sample_seed() + i)
                ind_lst.append(random.randint(0, len(self.train_dataset) - 1))
                random.seed()
            img_mds = [self.train_dataset.dataset[i] for i in ind_lst]
            samples = []
            for img_md in img_mds:
                _, _, bucket_size = self.train_dataset.get_size(img_md)
                sample = dict(
                    prompt=self.train_dataset.get_caption(img_md),
                    negative_prompt=self.train_dataset.get_negative_caption(img_md),
                    width=bucket_size[0],
                    height=bucket_size[1],
                    sample_name=f"train_{img_md['image_key']}",
                )
                if is_controlnet:
                    sample['control_image'] = self.train_dataset.get_control_image(img_md, type='pil')
                samples.append(sample)

            if self.valid_dataset is not None:
                ind_lst = []
                for i in range(self.eval_valid_size):
                    random.seed(self.get_sample_seed() + i)
                    ind_lst.append(random.randint(0, len(self.valid_dataset) - 1))
                    random.seed()
                for img_md in img_mds:
                    _, _, bucket_size = self.valid_dataset.get_size(img_md)
                    sample = dict(
                        prompt=self.valid_dataset.get_caption(img_md),
                        negative_prompt=self.valid_dataset.get_negative_caption(img_md),
                        width=bucket_size[0],
                        height=bucket_size[1],
                        sample_name=f"valid_{img_md['image_key']}",
                    )
                    if is_controlnet:
                        sample['control_image'] = self.valid_dataset.get_control_image(img_md, type='pil')
                    samples.append(sample)
            return samples
