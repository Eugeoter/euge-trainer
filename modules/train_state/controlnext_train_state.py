import os
import re
from typing import List, Union
from safetensors.torch import save_file
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils import eval_utils


class ControlNeXtTrainState(SD15TrainState):
    save_full_model: bool = False
    nnet_trainable_params: List[Union[str, re.Pattern]] = None

    def save_controlnext_model(self):
        self.logger.print(f"saving controlnext model at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_controlnext_ep{self.epoch}_step{self.global_step}.safetensors")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.unwrap_model(self.controlnext).save_pretrained(save_path, is_main_process=self.accelerator.is_main_process)
        controlnext = self.unwrap_model(self.controlnext)
        if self.accelerator.is_main_process:
            controlnext_sd = controlnext.state_dict()
            save_file(controlnext_sd, save_path)
        self.logger.print(f"controlnext model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def save_diffusion_model(self):
        if self.save_full_model:
            return super().save_diffusion_model()
        else:
            save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_nnet_ep{self.epoch}_step{self.global_step}.safetensors")
            if self.accelerator.is_main_process:
                nnet_sd = self.unwrap_model(self.nnet).state_dict()
                if isinstance(self.nnet_trainable_params, (str, re.Pattern)):
                    self.nnet_trainable_params = [self.nnet_trainable_params]
                if self.nnet_trainable_params is not None:
                    self.nnet_trainable_params = [re.compile(p) for p in self.nnet_trainable_params]
                    controlnext_nnet_sd = {k: v for k, v in nnet_sd.items() if any(p.match(k) for p in self.nnet_trainable_params)}
                else:
                    controlnext_nnet_sd = nnet_sd
                save_file(controlnext_nnet_sd, save_path)
            self.logger.print(f"controlnext nnet model saved to: `{logging.yellow(save_path)}`")
            return save_path

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            controlnet=self.unwrap_model(self.controlnext),
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )
