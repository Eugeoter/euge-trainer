import copy
import os
import torch.nn as nn
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils import eval_utils, device_utils, sd15_model_utils, lora_utils


class SD15LoRAAdapterTrainState(SD15TrainState):
    def save_diffusion_model(self) -> str:
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        nnet_with_merge = self.accelerator.unwrap_model(self.nnet_with_merge)
        text_encoder_with_merge = self.accelerator.unwrap_model(self.text_encoder_with_merge)
        for module_name, module in zip(self.lora_name_to_module_name.values(), self.lora_name_to_module.value()):
            module_with_merge = self.accelerator.unwrap_model(module)
            module_with_merge = module_with_merge.merged_module()
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                raise ValueError(f"LoRA module must be either nn.Linear or nn.Conv2d, got {type(module)}")
            lora_utils.set_module_by_name([nnet_with_merge, text_encoder_with_merge], module_name, module_with_merge)
        sd15_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.accelerator.unwrap_model(self.nnet_with_merge),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder_with_merge),
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

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet_with_merge),
            text_encoder=self.unwrap_model(self.text_encoder_with_merge),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )

    def get_original_pipeline(self):
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

    def get_with_lora_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet_with_lora),
            text_encoder=self.unwrap_model(self.text_encoder_with_lora),
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
        samples_with_lora = copy.deepcopy(samples)
        for sample_with_lora in samples_with_lora:
            sample_with_lora['sample_name'] = sample_with_lora.get('sample_name', eval_utils.DEFAULT_SAMPLE_NAME) + '_tau'
            sample_with_lora['prompt'] = ', '.join(self.lora_taus) + ', ' + sample_with_lora['prompt']
            samples.append(sample_with_lora)
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
                    original_pipeline = self.get_original_pipeline()
                    self.sample_images(original_pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                    self.logger.info(f"original sampled images saved to: `{logging.yellow(sample_dir)}`")
                    del original_pipeline

                    sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"lora")
                    lora_pipeline = self.get_with_lora_pipeline()
                    self.sample_images(lora_pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                    self.logger.info(f"lora sampled images saved to: `{logging.yellow(sample_dir)}`")
                    del lora_pipeline

                    device_utils.clean_memory()

                sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"ep{self.epoch}_step{self.global_step}")
                pipeline = self.get_pipeline()
                self.sample_images(pipeline, sample_dir, benchmark, on_epoch_end=on_epoch_end)
                self.logger.info(f"sampled images saved to: `{logging.yellow(sample_dir)}`")
            except Exception as e:
                import traceback
                self.logger.print(logging.red("exception when sample images:", e))
                traceback.print_exc()
                pass
        self.accelerator.wait_for_everyone()
