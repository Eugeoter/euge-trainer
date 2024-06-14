import os
from .train_state import TrainState
from ..utils import eval_utils, sdxl_model_utils, log_utils


class SDXLTrainState(TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        sdxl_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.unwrap_model(self.models.nnet),
            text_encoder=[self.unwrap_model(text_encoder) for text_encoder in self.models.text_encoder],
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_info=self.ckpt_info,
            vae=self.unwrap_model(self.models.vae),
            save_dtype=self.save_dtype,
            logit_scale=self.logit_scale,
        )
        self.logger.print(f"diffusion model saved to: `{log_utils.yellow(save_path)}`")

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.models.nnet),
            text_encoder=[self.unwrap_model(text_encoder) for text_encoder in self.models.text_encoder],
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.models.vae),
            scheduler=eval_utils.get_sampler(self.sample_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=self.clip_skip,
        )
