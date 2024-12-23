from .sd3_train_state import SD3TrainState
from .controlnext_train_state import ControlNeXtTrainState


class SD3ControlNeXtTrainState(SD3TrainState, ControlNeXtTrainState):
    def save_diffusion_model(self):
        return ControlNeXtTrainState.save_diffusion_model(self)

    def get_pipeline_psi(self):
        return self.pipeline_class(
            transformer=self.unwrap_model(self.nnet),
            controlnext=self.unwrap_model(self.controlnext),
            vae=self.unwrap_model(self.vae),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            text_encoder_3=self.unwrap_model(self.text_encoder[2]),
            tokenizer=self.tokenizer[0],
            tokenizer_2=self.tokenizer[1],
            tokenizer_3=self.tokenizer[2],
            scheduler=self.noise_scheduler,
        )
