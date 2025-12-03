from .sdxl_train_state import SDXLTrainState
from .sd15_inpainting_train_state import SD15InpaintingTrainState
from ..utils import eval_utils


class SDXLInpaintingTrainState(SDXLTrainState, SD15InpaintingTrainState):
    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder1),
            tokenizer=self.tokenizer1,
            text_encoder_2=self.unwrap_model(self.text_encoder2),
            tokenizer_2=self.tokenizer2,
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            feature_extractor=None,
        )
