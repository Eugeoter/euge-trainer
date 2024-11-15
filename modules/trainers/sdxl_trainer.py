import torch
import os
from safetensors.torch import load_file, save_file
from .sd15_trainer import SD15Trainer
from ..datasets.sdxl_dataset import SDXLDataset
from ..utils import train_utils, sdxl_model_utils, sdxl_train_utils, sd15_train_utils

from ..models.sdxl.nnet import SDXLUNet2DConditionModel
from ..pipelines.sdxl_lpw_pipeline import SDXLStableDiffusionLongPromptWeightingPipeline

# from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel as SDXLUNet2DConditionModel

from ..train_state.sdxl_train_state import SDXLTrainState


class SDXLTrainer(SD15Trainer):
    dataset_class = SDXLDataset
    nnet_class = SDXLUNet2DConditionModel
    pipeline_class = SDXLStableDiffusionLongPromptWeightingPipeline
    train_state_class = SDXLTrainState

    train_text_encoder: bool = False
    learning_rate_te1: float = None  # same as learning_rate by default
    learning_rate_te2: float = None  # same as learning_rate by default

    use_16c_vae: bool = False
    vae_adapter_model_name_or_path: str = None

    def check_config(self):
        super().check_config()
        if self.config.clip_skip is not None:
            self.logger.warning(f"clip_skip should not be used in sdxl models")

    def load_tokenizer_model(self):
        tokenizer1, tokenizer2 = sdxl_train_utils.load_sdxl_tokenizers(self.tokenizer_cache_dir or self.hf_cache_dir, self.max_token_length)
        return {'tokenizer1': tokenizer1, 'tokenizer2': tokenizer2}

    def load_diffusion_model(self):
        if os.path.isfile(self.pretrained_model_name_or_path):
            diffusion_models = sdxl_model_utils.load_models_from_sdxl_checkpoint(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                nnet_class=self.nnet_class,
            )
        else:
            diffusion_models = sdxl_model_utils.load_models_from_sdxl_diffusers_state(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                variant='fp16' if self.weight_dtype == torch.float16 else None,
                cache_dir=self.hf_cache_dir,
                token=self.hf_token,
                nnet_class=self.nnet_class,
                max_retries=self.max_retries,
            )
        return diffusion_models

    def load_vae_model(self):
        vae = super().load_vae_model()
        if self.use_16c_vae:
            from ..utils import lora_utils
            if not os.path.exists(self.vae_adapter_model_name_or_path):
                raise ValueError(f"16c VAE adapter checkpoint not found at {self.vae_adapter_model_name_or_path}")

            vae_adapter_sd = load_file(self.vae_adapter_model_name_or_path)

            lora_state_dict = {k: v for k, v in vae_adapter_sd.items() if "lora" in k}
            unet_state_dict = {k.replace("unet_", ""): v for k, v in vae_adapter_sd.items() if "unet_" in k}

            self.nnet.conv_in = torch.nn.Conv2d(16, 320, 3, 1, 1)
            self.nnet.conv_out = torch.nn.Conv2d(320, 16, 3, 1, 1)
            self.nnet.load_state_dict(unet_state_dict, strict=False)
            self.nnet.conv_in.to(self.weight_dtype)
            self.nnet.conv_out.to(self.weight_dtype)
            self.nnet.config.in_channels = 16
            self.nnet.config.out_channels = 16

            lora_utils.merge_loras_to_model(
                self.models,
                lora_state_dicts=[lora_state_dict],
                lora_ratios=[1.0],
                merge_device=self.nnet.device,
                merge_dtype=self.nnet.dtype,
                inplace=True,
            )

            # TODO: load lora state dict
            raise NotImplementedError("Loading LORA state dict is not implemented yet")

        return vae

    def get_vae_scale_factor(self):
        return sdxl_train_utils.VAE_SCALE_FACTOR

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

        # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
        if self.train_text_encoder1:
            self.text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
            self.text_encoder1.text_model.final_layer_norm.requires_grad_(False)

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
        return training_models, params_to_optimize

    def setup_block_lrs(self):
        assert len(self.block_lrs) == 23, f"block_lrs should have 23 values, got {len(self.block_lrs)}"

        block_params = [[] for _ in range(len(self.block_lrs))]

        for i, (name, param) in enumerate(self.nnet.named_parameters()):
            if name.startswith("time_embed.") or name.startswith("label_emb."):
                block_index = 0  # 0
            elif name.startswith("input_blocks."):  # 1-9
                block_index = 1 + int(name.split(".")[1])
            elif name.startswith("middle_block."):  # 10-12
                block_index = 10 + int(name.split(".")[1])
            elif name.startswith("output_blocks."):  # 13-21
                block_index = 13 + int(name.split(".")[1])
            elif name.startswith("out."):  # 22
                block_index = 22
            else:
                raise ValueError(f"unexpected parameter name: {name}")
            block_params[block_index].append(param)

        params_to_optimize = []
        for i, params in enumerate(block_params):
            if self.block_lrs[i] == 0:  # disable learning
                self.logger.info(f"disable learning for block {i}")
                continue
            params_to_optimize.append({"params": params, "lr": self.block_lrs[i]})
        return params_to_optimize

    def _print_start_training_message(self):
        super()._print_start_training_message()
        self.logger.print(f"  train nnet: {self.train_nnet} | learning rate: {self.learning_rate_nnet}")
        self.logger.print(f"  train text encoder 1: {self.train_text_encoder1} | learning rate: {self.learning_rate_te1}")
        self.logger.print(f"  train text encoder 2: {self.train_text_encoder2} | learning rate: {self.learning_rate_te2}")

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        # self.logger.debug(f"captions: {batch['captions']}")

        target_size = batch["target_size_hw"]
        orig_size = batch["original_size_hw"]
        crop_size = batch["crop_top_lefts"]
        text_embedding, vector_embedding = self.get_embeddings(batch['captions'], target_size, orig_size, crop_size, batch['negative_captions'] if self.do_classifier_free_guidance else None)
        text_embedding = text_embedding.to(self.weight_dtype)
        vector_embedding = vector_embedding.to(self.weight_dtype)

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(self.weight_dtype)

        with self.accelerator.autocast():
            model_pred = self.nnet(noisy_latents, timesteps, text_embedding, vector_embedding)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps, batch)
        return loss

    # def get_embeddings(self, captions, target_size, orig_size, crop_size, negative_captions=None):
    #     text_embeddings1, text_embeddings2, text_pool2, uncond_embeddings1, uncond_embeddings2, uncond_pool2 = self.encode_caption(captions, negative_captions)
    #     size_embeddings = sdxl_train_utils.get_size_embeddings(orig_size, crop_size, target_size, self.device)

    #     if self.do_classifier_free_guidance:
    #         text_embeddings = torch.cat([text_embeddings1, text_embeddings2], dim=2)
    #         uncond_embeddings = torch.cat([uncond_embeddings1, uncond_embeddings2], dim=2)
    #         text_embedding = torch.cat([uncond_embeddings, text_embeddings])

    #         cond_vector = torch.cat([text_pool2, size_embeddings], dim=1)
    #         uncond_vector = torch.cat([uncond_pool2, size_embeddings], dim=1)
    #         vector_embedding = torch.cat([uncond_vector, cond_vector])
    #     else:
    #         text_embedding = torch.cat([text_embeddings1, text_embeddings2], dim=2)
    #         vector_embedding = torch.cat([text_pool2, size_embeddings], dim=1)

    #     return text_embedding, vector_embedding

    # def encode_caption(
    #     self,
    #     captions,
    #     negative_captions=None,
    # ):
    #     with torch.set_grad_enabled(self.train_text_encoder1):
    #         text_embeddings1, text_pool1, uncond_embeddings1, uncond_pool1 = sdxl_train_utils.encode_caption(
    #             self.text_encoder1,
    #             self.tokenizer1,
    #             captions,
    #             negative_captions=negative_captions,
    #             max_embeddings_multiples=self.max_embeddings_multiples,
    #             clip_skip=self.clip_skip,
    #             do_classifier_free_guidance=self.do_classifier_free_guidance,
    #             is_sdxl_text_encoder2=False,
    #             skip_parsing=not self.caption_weighting,
    #         )
    #     with torch.set_grad_enabled(self.train_text_encoder2):
    #         text_embeddings2, text_pool2, uncond_embeddings2, uncond_pool2 = sdxl_train_utils.encode_caption(
    #             self.text_encoder2,
    #             self.tokenizer2,
    #             captions,
    #             negative_captions=negative_captions,
    #             max_embeddings_multiples=self.max_embeddings_multiples,
    #             clip_skip=self.clip_skip,
    #             do_classifier_free_guidance=self.do_classifier_free_guidance,
    #             is_sdxl_text_encoder2=True,
    #             skip_parsing=not self.caption_weighting,
    #         )
    #     return text_embeddings1, text_embeddings2, text_pool2, uncond_embeddings1, uncond_embeddings2, uncond_pool2

    def get_embeddings(self, captions, target_size, orig_size, crop_size, negative_captions=None):
        text_embeddings1, text_embeddings2, pool2 = self.encode_caption(captions, negative_captions)
        size_embeddings = sdxl_train_utils.get_size_embeddings(orig_size, crop_size, target_size, self.device)
        text_embedding = torch.cat([text_embeddings1, text_embeddings2], dim=2)
        vector_embedding = torch.cat([pool2, size_embeddings], dim=1)
        return text_embedding, vector_embedding

    def encode_caption(self, captions, negative_captions=None):
        input_ids1 = torch.stack([sd15_train_utils.get_input_ids(caption, self.tokenizer1, max_token_length=self.max_token_length) for caption in captions], dim=0)
        input_ids2 = torch.stack([sd15_train_utils.get_input_ids(caption, self.tokenizer2, max_token_length=self.max_token_length) for caption in captions], dim=0)
        with torch.set_grad_enabled(self.train_text_encoder):
            input_ids1 = input_ids1.to(self.device)
            input_ids2 = input_ids2.to(self.device)
            encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_utils.get_hidden_states_sdxl(
                self.max_token_length,
                input_ids1,
                input_ids2,
                self.tokenizer1,
                self.tokenizer2,
                self.text_encoder1,
                self.text_encoder2,
                None if not self.full_fp16 else self.weight_dtype,
            )
        return encoder_hidden_states1, encoder_hidden_states2, pool2
