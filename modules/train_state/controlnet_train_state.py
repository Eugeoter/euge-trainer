import os
import re
from waifuset import logging
from typing import List, Union
from safetensors.torch import save_file
from .sd15_train_state import SD15TrainState
from ..utils import eval_utils


class ControlNetTrainState(SD15TrainState):
    save_full_model: bool = False
    nnet_trainable_params: List[Union[str, re.Pattern]] = None

    def save_controlnet_model(self):
        self.logger.print(f"saving controlnet model at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_controlnet_ep{self.epoch}_step{self.global_step}.safetensors")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.unwrap_model(self.models.controlnet).save_pretrained(save_path, is_main_process=self.accelerator.is_main_process)
        controlnet = self.unwrap_model(self.controlnet)
        if self.accelerator.is_main_process:
            controlnet_sd = controlnet.state_dict()
            save_file(controlnet_sd, save_path)
        self.logger.print(f"controlnet model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def save_diffusion_model(self):
        return None

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            controlnet=self.unwrap_model(self.controlnet),
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )
