# Description: This file contains the temporary implementation of the FluxPipeline class used to generate images.

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union, List
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from ..utils import flux_train_utils, flux_model_utils, device_utils


class FluxPipeline:
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
    ):
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae = vae
        self.scheduler = scheduler
        self.transformer = transformer

        self.apply_t5_attn_mask = True
        self.device = None

    def to(self, device):
        if self.device == device:
            return self
        self.device = device
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)
        self.vae.to(device)
        self.transformer.to(device)
        return self

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        max_sequence_length=512,
    ):
        device_utils.clean_memory_on_device(self.device)
        org_vae_device = self.vae.device  # will be on cpu
        self.vae.to(self.device)  # distributed_state.device is same as device

        batch_size = len(prompt)
        prompt = prompt if isinstance(prompt, list) else [prompt or ""]

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size

        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        images = []
        for pmt, neg_pmt in zip(prompt, negative_prompt):

            tokens_and_masks = flux_train_utils.tokenize([self.tokenizer, self.tokenizer_2], pmt, t5xxl_max_token_length=max_sequence_length)
            # strategy has apply_t5_attn_mask option
            te_outputs = flux_train_utils.encode_tokens([self.text_encoder, self.text_encoder_2], tokens_and_masks, apply_t5_attn_mask=self.apply_t5_attn_mask)

            l_pooled, t5_out, txt_ids, t5_attn_mask = te_outputs

            # sample image
            weight_dtype = self.vae.dtype  # TOFO give dtype as argument
            packed_latent_height = height // 16
            packed_latent_width = width // 16

            img_ids = flux_model_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(self.device, weight_dtype)
            t5_attn_mask = t5_attn_mask.to(self.device) if self.apply_t5_attn_mask else None

            for i in range(num_images_per_prompt):
                noisy_latents = torch.randn(
                    1,
                    packed_latent_height * packed_latent_width,
                    16 * 2 * 2,
                    device=self.device,
                    dtype=weight_dtype,
                    generator=generator,
                )
                timesteps = flux_train_utils.get_schedule(num_inference_steps, noisy_latents.shape[1], shift=True)  # FLUX.1 dev -> shift=True

                with torch.no_grad():
                    guidance_vec = torch.full((noisy_latents.shape[0],), guidance_scale, device=noisy_latents.device, dtype=noisy_latents.dtype)
                    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
                        t_vec = torch.full((noisy_latents.shape[0],), t_curr, dtype=noisy_latents.dtype, device=noisy_latents.device)
                        pred = self.transformer(
                            img=noisy_latents,
                            img_ids=img_ids,
                            txt=t5_out,
                            txt_ids=txt_ids,
                            y=l_pooled,
                            timesteps=t_vec,
                            guidance=guidance_vec,
                            txt_attention_mask=t5_attn_mask,
                        )

                        noisy_latents = noisy_latents + (t_prev - t_curr) * pred

                noisy_latents = noisy_latents.float()
                noisy_latents = flux_model_utils.unpack_latents(noisy_latents, packed_latent_height, packed_latent_width)

                # latent to image
                with torch.no_grad():
                    noisy_latents = self.vae.decode(noisy_latents)

                noisy_latents = noisy_latents.clamp(-1, 1)
                noisy_latents = noisy_latents.permute(0, 2, 3, 1)
                image = Image.fromarray((127.5 * (noisy_latents + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

                images.append(image)

        self.vae.to(org_vae_device)
        device_utils.clean_memory_on_device(self.device)

        return FluxPipelineOutput(images=images)
