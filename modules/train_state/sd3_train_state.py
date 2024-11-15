import os
from .sd15_train_state import SD15TrainState
from waifuset import logging


class SD3TrainState(SD15TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}")
        pipeline = self.get_pipeline()
        pipeline.save_pretrained(save_path)
        self.logger.print(f"diffusion model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def get_pipeline(self):
        return self.pipeline_class(
            transformer=self.unwrap_model(self.nnet),
            vae=self.unwrap_model(self.vae),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            text_encoder_3=self.unwrap_model(self.text_encoder[2]),
            tokenizer=self.tokenizer[0],
            tokenizer_2=self.tokenizer[1],
            tokenizer_3=self.tokenizer[2],
            scheduler=self.noise_scheduler,
        )
