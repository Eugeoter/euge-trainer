import copy
import os
import torch.nn as nn
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils import eval_utils, device_utils, sd15_model_utils, lora_utils


class SD15LoRAAdaptationTrainState(SD15TrainState):
    def save_diffusion_model(self) -> str:
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")

        nnet_psi = copy.deepcopy(self.nnet.to('cpu'))
        text_encoder_psi = copy.deepcopy(self.text_encoder.to('cpu'))

        lora_name_to_save_module = lora_utils.make_lora_name_to_module_map([nnet_psi, text_encoder_psi], model_type=self.model_type)
        for lora_name in self.lora_name_to_module.keys():
            save_module = lora_name_to_save_module[lora_name]
            wrapper = self.lora_name_to_module[lora_name]
            wrapper = wrapper.to('cpu')
            merged_weight = wrapper.merged_weight()
            assert isinstance(save_module, (nn.Linear, nn.Conv2d)), f"save_module must be nn.Linear or nn.Conv2d, but got {save_module.__class__.__name__}"
            assert wrapper.module.__class__.__name__ == save_module.__class__.__name__, f"{wrapper.module.__class__.__name__} != {save_module.__class__.__name__}"
            assert wrapper.module.weight.data.shape == save_module.weight.data.shape, f"{wrapper.module.weight.data.shape} != {save_module.weight.data.shape}"
            assert merged_weight.shape == save_module.weight.data.shape, f"{merged_weight.shape} != {save_module.weight.data.shape}"
            save_module.weight.data = merged_weight
            wrapper = wrapper.to(self.device)

        sd15_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=nnet_psi,
            text_encoder=text_encoder_psi,
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

        # try:
        #     sd15_model_utils.load_models_from_stable_diffusion_checkpoint(save_path, device='cpu')
        #     self.logger.info(logging.green("Diffusion model loaded and tested successfully"))
        # except:
        #     raise

        # To original device
        self.nnet.to(self.device)
        self.text_encoder.to(self.device)

        return save_path

    def get_pipeline_psi(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet_psi),
            text_encoder=self.unwrap_model(self.text_encoder_psi),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )

    def get_pipeline_theta(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )

    def get_pipeline_phi(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet_phi),
            text_encoder=self.unwrap_model(self.text_encoder_phi),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )

    def get_benchmark(self):
        samples = super().get_benchmark()
        samples_phi = copy.deepcopy(samples)
        for sample_phi in samples_phi:
            sample_phi['sample_name'] = sample_phi.get('sample_name', eval_utils.DEFAULT_SAMPLE_NAME) + '_tau'
            sample_phi['prompt'] = ', '.join(self.taus_phi) + ', ' + sample_phi['prompt']
            samples.append(sample_phi)
        return samples

    def do_eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False, on_train_start=False):
        benchmark = self.get_benchmark()
        if not benchmark:
            self.logger.info("no benchmark found, skipping sample")
            return
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            try:
                if on_train_start:
                    sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"original")
                    original_pipeline = self.get_pipeline_theta()
                    self.sample_images(original_pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                    self.logger.info(f"original sampled images saved to: `{logging.yellow(sample_dir)}`")
                    del original_pipeline

                    sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"lora")
                    lora_pipeline = self.get_pipeline_phi()
                    self.sample_images(lora_pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                    self.logger.info(f"lora sampled images saved to: `{logging.yellow(sample_dir)}`")
                    del lora_pipeline

                    device_utils.clean_memory()

                sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"ep{self.epoch}_step{self.global_step}")
                pipeline = self.get_pipeline_psi()
                self.sample_images(pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                self.logger.info(f"sampled images saved to: `{logging.yellow(sample_dir)}`")
            except Exception as e:
                import traceback
                self.logger.print(logging.red("exception when sample images:", e))
                traceback.print_exc()
                pass
        self.accelerator.wait_for_everyone()
