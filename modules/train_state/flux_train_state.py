import os
from safetensors.torch import save_file
from waifuset import logging
from .sd15_train_state import SD15TrainState
from ..utils.sd15_model_utils import mem_eff_save_file


class FluxTrainState(SD15TrainState):
    train_nnet: bool = True
    train_text_encoder: bool = True

    def save_diffusion_model(self):
        save_dir = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}")

        if self.train_nnet or self.train_text_encoder[0] or self.train_text_encoder[1]:
            os.makedirs(save_dir, exist_ok=True)

        if self.train_nnet:
            nnet_save_path = os.path.join(save_dir, "nnet.safetensors")
            self.logger.info(f"saving nnet to {nnet_save_path}")
            nnet = self.accelerator.unwrap_model(self.nnet)
            nnet_sd = {}

            def update_sd(prefix, sd):
                for k, v in sd.items():
                    key = prefix + k
                    if self.save_dtype is not None and v.dtype != self.save_dtype:
                        v = v.detach().clone().to("cpu").to(self.save_dtype)
                    nnet_sd[key] = v

            update_sd("", nnet.state_dict())
            os.makedirs(self.output_model_dir, exist_ok=True)
            if not self.use_mem_eff_save:
                save_file(nnet_sd, nnet_save_path)
            else:
                mem_eff_save_file(nnet_sd, nnet_save_path)

        if self.train_text_encoder[0]:
            te1_save_path = os.path.join(save_dir, "text_encoder.safetensors")
            self.logger.info(f"saving CLIP L to {te1_save_path}")
            text_encoder = self.accelerator.unwrap_model(self.text_encoder[0])
            text_encoder_sd = {}

            def update_sd(prefix, sd):
                for k, v in sd.items():
                    key = prefix + k
                    if self.save_dtype is not None and v.dtype != self.save_dtype:
                        v = v.detach().clone().to("cpu").to(self.save_dtype)
                    text_encoder_sd[key] = v

            update_sd("", text_encoder.state_dict())
            os.makedirs(self.output_model_dir, exist_ok=True)
            if not self.use_mem_eff_save:
                save_file(text_encoder_sd, te1_save_path)
            else:
                mem_eff_save_file(text_encoder_sd, te1_save_path)

        if self.train_text_encoder[1]:
            te2_save_path = os.path.join(save_dir, "text_encoder_2.safetensors")
            self.logger.info(f"saving T5 XXL to {te2_save_path}")
            text_encoder_2 = self.accelerator.unwrap_model(self.text_encoder[1])
            text_encoder_2_sd = {}

            def update_sd(prefix, sd):
                for k, v in sd.items():
                    key = prefix + k
                    if self.save_dtype is not None and v.dtype != self.save_dtype:
                        v = v.detach().clone().to("cpu").to(self.save_dtype)
                    text_encoder_2_sd[key] = v

            update_sd("", text_encoder_2.state_dict())
            os.makedirs(self.output_model_dir, exist_ok=True)
            if not self.use_mem_eff_save:
                save_file(text_encoder_2_sd, te2_save_path)
            else:
                mem_eff_save_file(text_encoder_2_sd, te2_save_path)

        self.logger.print(f"diffusion model saved to: `{logging.yellow(save_dir)}`")
        return save_dir

    def get_pipeline_psi(self):
        return self.pipeline_class(
            transformer=self.unwrap_model(self.nnet),
            vae=self.unwrap_model(self.vae),
            text_encoder=self.unwrap_model(self.text_encoder[0]),
            text_encoder_2=self.unwrap_model(self.text_encoder[1]),
            tokenizer=self.tokenizer[0],
            tokenizer_2=self.tokenizer[1],
            scheduler=self.noise_scheduler,
        )
