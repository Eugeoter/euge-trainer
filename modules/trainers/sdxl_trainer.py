import torch
import os
import gc
from .t2i_trainer import T2ITrainer
from ..datasets.sdxl_dataset import SDXLTrainDataset
from ..utils import model_utils, train_utils, sdxl_model_utils, sdxl_train_utils
from ..models.nnet.sdxl_original_unet import SdxlUNet2DConditionModel
from ..pipelines.sdxl_lpw_pipeline import SdxlStableDiffusionLongPromptWeightingPipeline
from ..train_state.sdxl_train_state import SDXLTrainState


class SDXLT2ITrainer(T2ITrainer):
    dataset_class = SDXLTrainDataset
    nnet_class = SdxlUNet2DConditionModel
    pipeline_class = SdxlStableDiffusionLongPromptWeightingPipeline
    train_state_class = SDXLTrainState
    train_text_encoder: bool = False
    learning_rate_te1: float = None  # same as learning_rate
    learning_rate_te2: float = None  # same as learning_rate

    def load_diffusion_model(self):
        models = {}
        tokenizer1, tokenizer2 = sdxl_train_utils.load_sdxl_tokenizers(self.tokenizer_cache_dir or self.hf_cache_dir, self.max_token_length)
        models['tokenizer1'], models['tokenizer2'] = tokenizer1, tokenizer2
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
        models.update(diffusion_models)
        if self.vae_model_name_or_path is not None:
            models['vae'] = model_utils.load_vae(self.vae_model_name_or_path, dtype=self.weight_dtype)
            self.logger.print(f"additional vae model loaded from {self.vae_model_name_or_path}")
        return models

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
            self.learning_rate_te1
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
            self.learning_rate_te2
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)
        return training_models, params_to_optimize

    def get_train_state(self):
        train_state = self.train_state_class.from_config(
            self.config,
            self.accelerator,
            pipeline_class=self.pipeline_class,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_dataloader,
            save_dtype=self.save_dtype,
            nnet=self.nnet,
            text_encoder=[self.text_encoder1, self.text_encoder2],
            tokenizer=[self.tokenizer1, self.tokenizer2],
            vae=self.vae,
            logit_scale=self.logit_scale,
            ckpt_info=self.ckpt_info,
        )
        train_state.resume()
        return train_state

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

        input_ids1 = torch.stack([train_utils.get_input_ids(caption, self.tokenizer1, max_token_length=self.max_token_length) for caption in batch['captions']], dim=0)
        input_ids2 = torch.stack([train_utils.get_input_ids(caption, self.tokenizer2, max_token_length=self.max_token_length) for caption in batch['captions']], dim=0)
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

        target_size = batch["target_size_hw"]
        orig_size = batch["original_size_hw"]
        crop_size = batch["crop_top_lefts"]
        embs = sdxl_train_utils.get_size_embeddings(orig_size, crop_size, target_size, self.device).to(self.weight_dtype)

        vector_embedding = torch.cat([pool2, embs], dim=1).to(self.weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(self.weight_dtype)

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

        loss = self.get_loss(model_pred, target, timesteps)
        return loss
