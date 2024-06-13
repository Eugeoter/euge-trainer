import os
from .train_state import TrainState
from ..utils import eval_utils, sdxl_model_utils, log_utils


class SDXLTrainState(TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        sdxl_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.accelerator.unwrap_model(self.models['unet']),
            text_encoder=[self.accelerator.unwrap_model(text_encoder) for text_encoder in self.models['text_encoder']],
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_info=self.ckpt_info,
            vae=self.models['vae'],
            save_dtype=self.save_dtype,
            logit_scale=self.logit_scale,
        )
        self.logger.print(f"diffusion model saved to: `{log_utils.yellow(save_path)}`")
