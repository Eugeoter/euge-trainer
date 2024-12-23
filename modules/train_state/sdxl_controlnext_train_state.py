from .controlnext_train_state import ControlNeXtTrainState
from .sdxl_train_state import SDXLTrainState
from ..utils import eval_utils


class SDXLControlNeXtTrainState(ControlNeXtTrainState, SDXLTrainState):
    def get_pipeline_psi(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=[self.unwrap_model(self.text_encoder1), self.unwrap_model(self.text_encoder2)],
            tokenizer=[self.tokenizer1, self.tokenizer2],
            vae=self.unwrap_model(self.vae),
            controlnext=self.unwrap_model(self.controlnext),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=self.clip_skip,
        )
