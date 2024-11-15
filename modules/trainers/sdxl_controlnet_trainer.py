import torch
import random
import numpy as np
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from .sd15_controlnet_trainer import SD15ControlNetTrainer
from .sdxl_trainer import SDXLTrainer
# from ..models.sdxl.nnet import SDXLUNet2DConditionModel
# from ..pipelines.sdxl_lpw_pipeline import SDXLStableDiffusionLongPromptWeightingPipeline
from ..train_state.sdxl_controlnet_train_state import SDXLControlNetTrainState
from ..datasets.sdxl_controlnet_dataset import SDXLControlNetDataset
from ..utils import sdxl_model_utils, sdxl_train_utils


class SDXLControlNetTrainer(SDXLTrainer, SD15ControlNetTrainer):
    nnet_param_names = None
    dataset_class = SDXLControlNetDataset
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

    def get_embeddings(self, captions, target_size, orig_size, crop_top_lefts, negative_captions=None):
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            captions,
            [self.text_encoder1, self.text_encoder2],
            [self.tokenizer1, self.tokenizer2],
            proportion_empty_prompts=0,
            is_train=True,
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = torch.cat([orig_size, crop_top_lefts, target_size], dim=1).to(self.device)
        # self.logger.info(f"add_time_ids: {add_time_ids}, shape: {add_time_ids.shape}")
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        if torch.isnan(latents).any():
            raise ValueError("Latents contain NaNs")
        latents *= self.vae_scale_factor

        target_size = batch["target_size_hw"]
        orig_size = batch["original_size_hw"]
        crop_top_lefts = batch["crop_top_lefts"]
        prompt_embeds, unet_added_conditions = self.get_embeddings(
            batch['captions'],
            target_size,
            orig_size,
            crop_top_lefts,
        )

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        control_images = batch['control_images'].to(self.device, dtype=self.controlnet.dtype)

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


def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):  # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
