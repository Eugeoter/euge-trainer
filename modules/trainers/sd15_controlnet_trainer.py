import torch
import os
from torch import nn
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from .sd15_trainer import SD15Trainer
from ..train_state.controlnet_train_state import ControlNetTrainState
from ..datasets.controlnet_dataset import ControlNetDataset
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline


class SD15ControlNetTrainer(SD15Trainer):
    nnet_param_names = None
    dataset_class = ControlNetDataset
    nnet_class = UNet2DConditionModel
    pipeline_class = StableDiffusionControlNetPipeline
    train_state_class = ControlNetTrainState
    controlnet_class = ControlNetModel
    controlnet_model_name_or_path: str = None

    train_controlnet: bool = True
    learning_rate_controlnet: float = None

    def check_config(self):
        super().check_config()
        if not self.train_controlnet:
            self.logger.warning("controlnet is not being trained!")

    def get_model_loaders(self):
        loaders = super().get_model_loaders()
        loaders.remove(self.load_controlnet_model)
        loaders.append(self.load_controlnet_model)
        return loaders

    def load_controlnet_model(self):
        if self.controlnet_model_name_or_path is not None:
            self.logger.info(f"Loading controlnet model from {self.controlnet_model_name_or_path}")
            if os.path.isfile(self.controlnet_model_name_or_path):
                controlnet = self.controlnet_class.from_single_file(
                    self.controlnet_model_name_or_path,
                    use_safetensors=os.path.splitext(self.controlnet_model_name_or_path)[1] == ".safetensors",
                )
            else:
                controlnet = self.controlnet_class.from_pretrained(self.controlnet_model_name_or_path)
            self.logger.info(f"Controlnet model loaded from {self.controlnet_model_name_or_path}")
        else:
            controlnet = self.controlnet_class.from_unet(self.nnet)
            self.logger.info(f"Controlnet model initialized from nnet")
        return {"controlnet": controlnet}

    def enable_xformers(self):
        super().enable_xformers()
        self.controlnet.enable_xformers_memory_efficient_attention()

    def setup_controlnet_params(self):
        training_models = []
        params_to_optimize = []

        lr_controlnet = self.learning_rate_controlnet or self.learning_rate
        # self.logger.debug(f"lr_controlnet: {lr_controlnet}")
        train_controlnet = self.train_controlnet and lr_controlnet > 0
        if train_controlnet:
            if self.gradient_checkpointing:
                try:
                    self.controlnet.enable_gradient_checkpointing()
                except AttributeError:
                    self.logger.warning(f"Gradient checkpointing is not supported for {self.controlnet.__class__.__name__}")
                else:
                    self.logger.info("Gradient checkpointing enabled for controlnet")
            self.controlnet.train()
            training_models.append(self.controlnet)
            params_to_optimize.append({"params": list(self.controlnet.parameters()), "lr": lr_controlnet})
        else:
            lr_controlnet = 0
            self.controlnet.eval()
        self.controlnet.requires_grad_(train_controlnet)
        self.controlnet = self._prepare_one_model(self.controlnet, train=train_controlnet, dtype=torch.float32, transform_model_if_ddp=True)
        self.train_controlnet = train_controlnet

        assert self.accelerator.unwrap_model(self.controlnet).dtype == torch.float32, f"Expected controlnet dtype to be torch.float32, but got {self.accelerator.unwrap_model(self.controlnet).dtype}"

        return training_models, params_to_optimize

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
    #         controlnet=self.controlnet,
    #         text_encoder=self.text_encoder,
    #         tokenizer=self.tokenizer,
    #         vae=self.vae,
    #     )

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        encoder_hidden_states = self.encode_caption(batch['captions'])

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(latents.dtype)

        control_images = batch['control_images'].to(self.device, dtype=torch.float32)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_images,
            return_dict=False,
        )

        with self.accelerator.autocast():
            model_pred = self.nnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
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
