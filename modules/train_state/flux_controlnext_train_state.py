from .flux_train_state import FluxTrainState
from .controlnext_train_state import ControlNeXtTrainState


class FluxControlNeXtTrainState(FluxTrainState, ControlNeXtTrainState):
    def save_diffusion_model(self):
        return ControlNeXtTrainState.save_diffusion_model(self)

    def get_pipeline_psi(self):
        return self.pipeline_class(
            transformer=self.unwrap_model(self.nnet),
            vae=self.unwrap_model(self.vae),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            tokenizer=self.tokenizer[0],
            tokenizer_2=self.tokenizer[1],
            scheduler=self.noise_scheduler,
            controlnet=self.unwrap_model(self.controlnext),
        )
