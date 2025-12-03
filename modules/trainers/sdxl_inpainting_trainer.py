import torch
import random
import numpy as np
from torch import nn
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from .sdxl_trainer import SDXLTrainer
# from ..models.sdxl.nnet import SDXLUNet2DConditionModel
# from ..pipelines.sdxl_lpw_pipeline import SDXLStableDiffusionLongPromptWeightingPipeline
from ..train_state.sdxl_inpainting_train_state import SDXLInpaintingTrainState
from ..datasets.sdxl_image_condition_dataset import SDXLImageConditionDataset
from ..utils import sdxl_model_utils, sdxl_train_utils
from .sdxl_trainer import SDXLTrainer


class SDXLInpaintingTrainer(SDXLTrainer):
    nnet_param_names = None
    dataset_class = SDXLImageConditionDataset
    nnet_class = UNet2DConditionModel
    pipeline_class = StableDiffusionXLInpaintPipeline
    train_state_class = SDXLInpaintingTrainState

    def load_diffusion_model(self):
        models = sdxl_model_utils.load_diffusers_models(
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

        nnet = models["nnet"]
        pretrained_weight = nnet.conv_in.weight  # shape: [block_out_channels[0], 4, kernel_size, kernel_size]
        if pretrained_weight.shape[1] == 4:  # in_channels == 4
            self.logger.info(f"Converting pretrained model to inpainting model")
            pretrained_bias = nnet.conv_in.bias     # shape: [block_out_channels[0]]

            new_weight = torch.zeros(nnet.config.block_out_channels[0], 9, nnet.config.conv_in_kernel, nnet.config.conv_in_kernel, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            new_weight[:, :4, :, :] = pretrained_weight

            nnet.conv_in = nn.Conv2d(
                in_channels=9,
                out_channels=nnet.config.block_out_channels[0],
                kernel_size=nnet.config.conv_in_kernel,
                padding=(nnet.config.conv_in_kernel - 1) // 2,
                dtype=pretrained_weight.dtype,
                device=pretrained_weight.device
            )

            nnet.conv_in.weight.data = new_weight
            nnet.conv_in.bias.data = pretrained_bias
            nnet.config.in_channels = 9  # overwrite in_channels in config
            models["nnet"] = nnet
        else:
            self.logger.info(f"Using pretrained 9-channel model")
        return models

    def train_step(self, batch):
        pixel_values = batch["images"].to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        # if torch.isnan(latents).any():
        #     raise ValueError("Latents contain NaNs")
        latents *= self.vae_scale_factor

        masks_orig = batch["condition_images"].to(self.device)
        masks_orig /= 255.0  # Normalize to [0, 1]

        # Prepare masked latents
        masked_pixels = pixel_values * (1.0 - masks_orig)
        with torch.no_grad():
            masked_latents = self.vae.encode(masked_pixels.to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        masked_latents = masked_latents * self.vae.config.scaling_factor

        # Prepare mask
        vae_downscaling_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # == 8
        mask = torch.nn.functional.interpolate(
            masks_orig,
            size=(
                int(masks_orig.shape[2] // vae_downscaling_factor),
                int(masks_orig.shape[3] // vae_downscaling_factor)
            )
        )
        mask = mask[:, 0:1]
        mask = mask.to(self.device, dtype=self.weight_dtype)

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

        combined_latents = torch.cat([noisy_latents, mask, masked_latents], dim=1)

        with self.accelerator.autocast():
            model_pred = self.nnet(
                combined_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
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
