import os
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils import sdxl_model_utils


class SDXLTrainState(SD15TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        sdxl_model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.unwrap_model(self.nnet),
            text_encoder=[self.unwrap_model(self.text_encoder1), self.unwrap_model(self.text_encoder2)],
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_info=self.ckpt_info,
            vae=self.unwrap_model(self.vae),
            save_dtype=self.save_dtype,
            logit_scale=self.logit_scale,
            use_mem_eff_save=self.use_mem_eff_save,
            v_pred=self.prediction_type == 'v_prediction',
            ztsnr=self.zero_terminal_snr,
        )
        self.logger.print(f"diffusion model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=[self.unwrap_model(self.text_encoder1), self.unwrap_model(self.text_encoder2)],
            tokenizer=[self.tokenizer1, self.tokenizer2],
            vae=self.unwrap_model(self.vae),
            scheduler=self.get_sampler(),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=self.clip_skip,
        )
