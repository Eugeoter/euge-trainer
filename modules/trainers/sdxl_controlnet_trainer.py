import torch
import random
import numpy as np
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from .sd15_controlnet_trainer import SD15ControlNetTrainer
from .sdxl_trainer import SDXLTrainer
# from ..models.sdxl.nnet import SDXLUNet2DConditionModel
# from ..pipelines.sdxl_lpw_pipeline import SDXLStableDiffusionLongPromptWeightingPipeline
from ..train_state.sdxl_controlnet_train_state import SDXLControlNetTrainState
from ..datasets.sdxl_image_condition_dataset import SDXLImageConditionDataset
from ..utils import sdxl_model_utils, sdxl_train_utils


class SDXLControlNetTrainer(SDXLTrainer, SD15ControlNetTrainer):
    nnet_param_names = None
    dataset_class = SDXLImageConditionDataset
    nnet_class = UNet2DConditionModel
    pipeline_class = StableDiffusionXLControlNetPipeline
    train_state_class = SDXLControlNetTrainState
    controlnet_class = ControlNetModel

    def load_diffusion_model(self):
        return sdxl_model_utils.load_diffusers_models(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            variant=self.variant,
            torch_dtype=self.weight_dtype,
            use_safetensors=self.use_safetensors,
            cache_dir=self.hf_cache_dir,
            token=self.hf_token,
            max_retries=self.max_retries,
            nnet_class=self.nnet_class,
        )

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        # if torch.isnan(latents).any():
        #     raise ValueError("Latents contain NaNs")
        latents *= self.vae_scale_factor

        target_size = batch["target_size_hw"]
        orig_size = batch["original_size_hw"]
        crop_top_lefts = batch["crop_top_lefts"]
        prompt_embeds, unet_added_conditions = self.get_embeddings_diffusers(
            batch['captions'],
            target_size,
            orig_size,
            crop_top_lefts,
        )

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        control_images = batch['condition_images'].to(self.device, dtype=self.controlnet.dtype)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            controlnet_cond=control_images,
            return_dict=False,
        )

        with self.accelerator.autocast():
            model_pred = self.nnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                return_dict=False,
            )[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps, batch)
        return loss
