import torch
from .sd3_trainer import SD3Trainer
from .sd15_controlnext_trainer import SD15ControlNeXtTrainer
from ..datasets.controlnet_dataset import ControlNetDataset
from ..models.sd3.controlnext import SD3ControlNeXtModel
from ..models.sd3.controlnext_nnet import SD3ControlAnyTransformer2DModel
from ..train_state.sd3_controlnext_train_state import SD3ControlNeXtTrainState
from ..pipelines.sd3_controlnext_pipeline import StableDiffusion3ControlNeXtPipeline


class SD3ControlAnyTrainer(SD3Trainer, SD15ControlNeXtTrainer):
    dataset_class = ControlNetDataset
    train_state_class = SD3ControlNeXtTrainState
    nnet_class = SD3ControlAnyTransformer2DModel
    pipeline_class = StableDiffusion3ControlNeXtPipeline
    controlnext_class = SD3ControlNeXtModel

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
    #         text_encoder=[self.text_encoder1, self.text_encoder2, self.text_encoder3],
    #         tokenizer=[self.tokenizer1, self.tokenizer2, self.tokenizer3],
    #         noise_scheduler=self.noise_scheduler,
    #         vae=self.vae,
    #     )

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
        noisy_latents = self.get_noisy_latents(latents, noise, sigmas)
        noisy_latents = noisy_latents.to(self.weight_dtype)

        control_images = batch['control_images'].to(self.device, dtype=self.weight_dtype)
        controlnext_output = self.controlnext(control_images, timesteps, return_dict=False)

        with self.accelerator.autocast():
            model_pred = self.nnet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
                control_signal=controlnext_output,
            )[0]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_latents

        target = latents
        weighting = self.get_loss_weighting(sigmas, bsz)
        loss = self.get_loss(model_pred, target, weighting)
        return loss
