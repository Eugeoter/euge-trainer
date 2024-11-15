from .hunyuan_train_state import HunyuanTrainState
from .controlnext_train_state import ControlNeXtTrainState
from ..utils import hunyuan_model_utils


class HunyuanControlNeXtTrainState(HunyuanTrainState, ControlNeXtTrainState):

    def save_diffusion_model(self):
        return ControlNeXtTrainState.save_diffusion_model(self)

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            controlnet=self.unwrap_model(self.controlnext),
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=self.unwrap_model(self.vae),
            scheduler=hunyuan_model_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
