import os
from .train_state import TrainState
from ..utils import log_utils


class SD3TrainState(TrainState):
    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        pipeline = self.get_pipeline()
        pipeline.save_pretrained(save_path)
        self.logger.print(f"diffusion model saved to: `{log_utils.yellow(save_path)}`")

    def get_pipeline(self):
        return self.pipeline_class(
            transformer=self.unwrap_model(self.models.nnet),
            vae=self.unwrap_model(self.models.vae),
            text_encoder=self.unwrap_model(self.models.text_encoder[0]),
            text_encoder_2=self.unwrap_model(self.models.text_encoder[1]),
            text_encoder_3=self.unwrap_model(self.models.text_encoder[2]),
            tokenizer=self.tokenizer[0],
            tokenizer_2=self.tokenizer[1],
            tokenizer_3=self.tokenizer[2],
            scheduler=self.models.noise_scheduler,
        )
