import torch
from torch import nn
from .sd15_trainer import SD15Trainer
from ..train_state.controlnext_train_state import ControlNeXtTrainState
from ..datasets.controlnet_dataset import ControlNetDataset
from ..models.sd15.controlnext_nnet import ControlNeXtUNet2DConditionModel
from ..models.sd15.controlnext import ControlNeXtModel
from ..pipelines.controlnext_pipeline import StableDiffusionControlNeXtPipeline


class SD15ControlNeXtTrainer(SD15Trainer):
    nnet_param_names = None
    dataset_class = ControlNetDataset
    nnet_class = ControlNeXtUNet2DConditionModel
    pipeline_class = StableDiffusionControlNeXtPipeline
    train_state_class = ControlNeXtTrainState

    controlnext_class = ControlNeXtModel
    controlnext_model_name_or_path: str = None

    train_controlnext: bool = True
    control_scale: float = 1.0
    learning_rate_controlnext: float = None
    learn_control_scale: bool = False
    initial_control_scale: float = 1.0

    def check_config(self):
        super().check_config()
        if not self.train_controlnext:
            self.logger.warning("ControlNeXt is not being trained.")

    def load_controlnext_model(self):
        if self.controlnext_model_name_or_path is not None:
            controlnext = self.controlnext_class.from_pretrained(self.controlnext_model_name_or_path)
            self.logger.info(f"ControlNeXt model loaded from {self.controlnext_model_name_or_path}")
        else:
            controlnext = self.controlnext_class()
            self.logger.info(f"ControlNeXt model initialized from scratch")
        if self.learn_control_scale:
            controlnext.scale = nn.Parameter(torch.tensor(self.initial_control_scale), requires_grad=True)
            self.logger.info(f"ControlNeXt model scale is set to learnable and initialized to {controlnext.scale.item()}")
        return {"controlnext": controlnext}

    def enable_xformers(self):
        super().enable_xformers()
        self.controlnext.enable_xformers_memory_efficient_attention()

    def setup_controlnext_params(self):
        training_models = []
        params_to_optimize = []

        lr_controlnext = self.learning_rate_controlnext or self.learning_rate
        # self.logger.debug(f"lr_controlnext: {lr_controlnext}")
        train_controlnext = self.train_controlnext and lr_controlnext > 0
        if train_controlnext:
            if self.gradient_checkpointing:
                try:
                    self.controlnext.enable_gradient_checkpointing()
                except AttributeError:
                    self.logger.warning(f"Gradient checkpointing is not supported for {self.controlnext.__class__.__name__}")
            self.controlnext.train()
            training_models.append(self.controlnext)
            params_to_optimize.append({"params": list(self.controlnext.parameters()), "lr": lr_controlnext})
        else:
            lr_controlnext = 0
            self.controlnext.eval()
        self.controlnext.requires_grad_(train_controlnext)
        self.controlnext = self._prepare_one_model(self.controlnext, dtype=torch.float32, train=train_controlnext, transform_model_if_ddp=True)
        self.train_controlnext = train_controlnext

        assert self.controlnext.dtype == torch.float32, f"Expected controlnext dtype to be torch.float32, but got {self.controlnext.dtype}"

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
    #         controlnext=self.controlnext,
    #         text_encoder=self.text_encoder,
    #         tokenizer=self.tokenizer,
    #         vae=self.vae,
    #     )

    def get_start_training_message(self):
        msg = super().get_start_training_message()
        msg.append(f"fixed control scale: {self.control_scale}")
        msg.append(f"learn control scale: {self.learn_control_scale}")
        if self.learn_control_scale:
            msg.append(f"initial control scale: {self.initial_control_scale}")
        return msg

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

        control_images = batch['control_images'].to(self.device, dtype=self.controlnext.dtype)
        controls = self.controlnext(control_images, timesteps)
        controls['scale'] = controls['scale'] * self.control_scale

        with self.accelerator.autocast():
            model_pred = self.nnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=None,  # [sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=None,  # mid_block_res_sample.to(dtype=weight_dtype),
                controls=controls,
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
