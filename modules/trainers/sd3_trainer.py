import copy
import math
import os
import torch
from typing import Literal
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from .sd15_trainer import SD15Trainer
from ..train_state.sd3_train_state import SD3TrainState
from ..datasets.t2i_dataset import T2IDataset
from ..utils import sd3_model_utils


class SD3Trainer(SD15Trainer):
    dataset_class = T2IDataset
    train_state_class = SD3TrainState
    nnet_class = SD3Transformer2DModel
    pipeline_class = StableDiffusion3Pipeline
    dropout_t5: bool = True
    timestep_sampler_type: Literal['sigma_sqrt', 'logit_normal', 'mode'] = "logit_normal"

    def load_tokenizer_model(self):
        # tokenizers are loaded from `load_diffusion_model`
        return {}

    def load_diffusion_model(self):
        # tokenizer1, tokenizer2, tokenizer3 = sd3_model_utils.load_sd3_tokenizers(
        #     self.tokenizer_cache_dir,
        #     max_token_length=self.max_token_length
        # )
        # models['tokenizer1'], models['tokenizer2'], models['tokenizer3'] = tokenizer1, tokenizer2, tokenizer3
        if os.path.isfile(self.pretrained_model_name_or_path):
            diffusion_models = sd3_model_utils.load_models_from_stable_diffusion_checkpoint(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=torch.float16,
                nnet_class=self.nnet_class,
            )

        else:
            diffusion_models = sd3_model_utils.load_models_from_stable_diffusion_diffusers_state(
                self.pretrained_model_name_or_path,
                revision=self.revision,
                variant=self.variant,
                device=self.device,
                dtype=torch.float16,
                cache_dir=self.hf_cache_dir,
                nnet_class=self.nnet_class,
                dropout_t5=self.dropout_t5,
                max_retries=self.max_retries,
            )
        return diffusion_models

    def get_vae_scale_factor(self):
        return self.vae.config.scaling_factor

    def enable_xformers(self):
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("Please install xformers to use the xformers model")
        # self.nnet.set_use_memory_efficient_attention_xformers(True, False)
        if torch.__version__ >= "2.0.0":
            self.vae.set_use_memory_efficient_attention_xformers(True)

    def setup_text_encoder_params(self):
        training_models = []
        params_to_optimize = []

        (
            training_models_,
            params_to_optimize_,
            self.text_encoder1,
            self.train_text_encoder1,
            self.learning_rate_te1
        ) = self._setup_one_text_encoder_params(
            self.text_encoder1,
            self.learning_rate_te1,
            name="text_encoder1"
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)

        (
            training_models_,
            params_to_optimize_,
            self.text_encoder2,
            self.train_text_encoder2,
            self.learning_rate_te2
        ) = self._setup_one_text_encoder_params(
            self.text_encoder2,
            self.learning_rate_te2,
            name="text_encoder2"
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)

        if not self.dropout_t5:
            (
                training_models_,
                params_to_optimize_,
                self.text_encoder3,
                self.train_text_encoder3,
                self.learning_rate_te3
            ) = self._setup_one_text_encoder_params(
                self.text_encoder3,
                self.learning_rate_te3,
                name="text_encoder3",
            )
            training_models.extend(training_models_)
            params_to_optimize.extend(params_to_optimize_)
        else:
            self.text_encoder3 = None
            self.tokenizer3 = None
            self.train_text_encoder3 = False
            self.learning_rate_te3 = 0
        return training_models, params_to_optimize

    def get_noise_scheduler(self):
        if hasattr(self, "noise_scheduler"):
            noise_scheduler = self.noise_scheduler
        else:
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="scheduler"
            )
        self.noise_scheduler_ = copy.deepcopy(noise_scheduler)  # copy
        return noise_scheduler

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # def get_train_state(self):
    #     return self.train_state_class.from_config(
    #         self.config,
    #         self,
    #         self.accelerator,
    #         train_dataset=self.train_dataset,
    #         valid_dataset=self.valid_dataset,
    #         pipeline_class=self.pipeline_class,
    #         optimizer=self.optimizer,
    #         lr_scheduler=self.lr_scheduler,
    #         train_dataloader=self.train_dataloader,
    #         valid_dataloader=self.valid_dataloader,
    #         save_dtype=self.save_dtype,
    #         nnet=self.nnet,
    #         text_encoder=[self.text_encoder1, self.text_encoder2, self.text_encoder3],
    #         tokenizer=[self.tokenizer1, self.tokenizer2, self.tokenizer3],
    #         noise_scheduler=self.noise_scheduler,
    #         vae=self.vae,
    #     )

    def _print_start_training_message(self):
        super()._print_start_training_message()
        self.logger.print(f"  train nnet: {self.train_nnet} | learning rate: {self.learning_rate_nnet}")
        self.logger.print(f"  train text encoder 1: {self.train_text_encoder1} | learning rate: {self.learning_rate_te1}")
        self.logger.print(f"  train text encoder 2: {self.train_text_encoder2} | learning rate: {self.learning_rate_te2}")
        if not self.dropout_t5:
            self.logger.print(f"  train text encoder 3: {self.train_text_encoder3} | learning rate: {self.learning_rate_te3}")
        else:
            self.logger.print("  text encoder 3 dropped")

    def get_timesteps(self, latents):
        b_size = latents.shape[0]
        indices = torch.randint(0, self.noise_scheduler_.config.num_train_timesteps, (b_size,))
        timesteps = self.noise_scheduler_.timesteps[indices].to(device=latents.device)
        return timesteps

    def get_noisy_latents(self, latents, noise, sigmas):
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        return noisy_latents

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_.timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        with torch.set_grad_enabled(self.train_text_encoder):
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=batch['captions'], device=self.device)
            if self.full_fp16:
                prompt_embeds = prompt_embeds.to(self.weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)

        bsz = latents.shape[0]
        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents).to(self.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = self.get_noisy_latents(latents, noise, sigmas).to(self.weight_dtype)

        with self.accelerator.autocast():
            model_pred = self.nnet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_latents

        target = latents
        weighting = self.get_loss_weighting(sigmas, bsz)
        loss = self.get_loss(model_pred, target, weighting)
        return loss

    def _encode_prompt_with_t5(
        self,
        tokenizer,
        text_encoder,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if text_encoder is None:
            return torch.zeros(
                (batch_size, self.max_token_length, self.nnet.config.joint_attention_dim),
                device=device,
                dtype=self.weight_dtype,
            )

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = [self.tokenizer1, self.tokenizer2]
        clip_text_encoders = [self.text_encoder1, self.text_encoder2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device if device is not None else text_encoder.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = self._encode_prompt_with_t5(
            tokenizer=self.tokenizer3,
            text_encoder=self.text_encoder3,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else self.text_encoder3.device if self.text_encoder3 is not None else None,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds

    def get_loss_weighting(self, sigmas, bsz):
        # TODO (kashif, sayakpaul): weighting sceme needs to be experimented with :)
        if self.timestep_sampler_type == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif self.timestep_sampler_type == "logit_normal":
            timestep_sampler_kwargs = self.timestep_sampler_kwargs
            m = timestep_sampler_kwargs.get('loc', 0) or timestep_sampler_kwargs.get('mean', 0) or timestep_sampler_kwargs.get('m', 0) or timestep_sampler_kwargs.get('mu', 0)
            s = timestep_sampler_kwargs.get('scale', 1) or timestep_sampler_kwargs.get('std', 1) or timestep_sampler_kwargs.get('s', 1) or timestep_sampler_kwargs.get('sigma', 1)
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=m, std=s, size=(bsz,), device=self.device)
            weighting = torch.nn.functional.sigmoid(u)
        elif self.timestep_sampler_type == "mode":
            # See sec 3.1 in the SD3 paper (20).
            u = torch.rand(size=(bsz,), device=self.device)
            weighting = 1 - u - self.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            raise ValueError(f"Unknown timestep sampler type {self.timestep_sampler_type}")
        return weighting

    def get_loss(self, model_pred, target, weighting):
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss
