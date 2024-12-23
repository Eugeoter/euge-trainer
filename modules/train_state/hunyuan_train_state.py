import os
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils import hunyuan_model_utils


class HunyuanTrainState(SD15TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}")
        hunyuan_model_utils.save_hunyuan_checkpoint(
            output_dir=save_path,
            model=self.unwrap_model(self.get_nnet()),
            text_encoder1=self.unwrap_model(self.text_encoder[0]),
            text_encoder2=self.unwrap_model(self.text_encoder[1]),
            ema=self.nnet_ema if hasattr(self, "nnet_ema") else None,
            epoch=self.epoch,
            step=self.global_step,
        )
        self.logger.print(f"diffusion model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def get_pipeline_psi(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.get_nnet()),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=self.unwrap_model(self.vae),
            scheduler=hunyuan_model_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    def get_nnet(self):
        from ..models.hunyuan.modules.fp16_layers import Float16Module
        return self.nnet if not isinstance(self.nnet, Float16Module) else self.nnet.module
