import torch
from .sd15_controlnext_trainer import SD15ControlNeXtTrainer
from .sdxl_trainer import SDXLTrainer
from ..utils import sdxl_train_utils

from ..models.sdxl.controlnext_nnet import SDXLControlNeXtUNet2DConditionModel
from ..models.sdxl.controlnext import ControlNetModel
from ..pipelines.sdxl_controlnext_lpw_pipeline import StableDiffusionXLControlNeXtPipeline

# from ..models.sd15.controlnext import ControlNetModel
# from ..models.sd15.controlnext_nnet_pbh import UNet2DConditionModel as SDXLControlNeXtUNet2DConditionModel
# from ..pipelines.sdxl_controlnext_pipeline import StableDiffusionXLControlNeXtPipeline

from ..datasets.sdxl_controlnet_dataset import SDXLControlNetDataset
from ..train_state.sdxl_controlnext_train_state import SDXLControlNeXtTrainState


class SDXLControlNeXtTrainer(SD15ControlNeXtTrainer, SDXLTrainer):
    dataset_class = SDXLControlNetDataset
    nnet_class = SDXLControlNeXtUNet2DConditionModel
    controlnext_class = ControlNetModel
    pipeline_class = StableDiffusionXLControlNeXtPipeline
    train_state_class = SDXLControlNeXtTrainState

    control_scale: float = 1.0

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
    #         text_encoder=[self.text_encoder1, self.text_encoder2],
    #         tokenizer=[self.tokenizer1, self.tokenizer2],
    #         vae=self.vae,
    #         ckpt_info=self.ckpt_info,
    #         logit_scale=self.logit_scale,
    #     )

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        target_size = batch["target_size_hw"]
        orig_size = batch["original_size_hw"]
        crop_size = batch["crop_top_lefts"]
        text_embedding, vector_embedding = self.get_embeddings(batch['captions'], target_size, orig_size, crop_size, batch['negative_captions'] if self.do_classifier_free_guidance else None)
        text_embedding = text_embedding.to(self.weight_dtype)
        vector_embedding = vector_embedding.to(self.weight_dtype)

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        control_images = batch['control_images'].to(self.device, dtype=self.controlnext.dtype)
        controls = self.controlnext(control_images, timesteps)
        controls['scale'] = controls['scale'] * self.control_scale
        # added_cond_kwargs = {'text_embeds': pool2, 'time_ids': vector_embedding}

        # self.logger.debug(f"text_embedding: {text_embedding.shape}, vector_embedding: {vector_embedding.shape}")

        with self.accelerator.autocast():
            model_pred = self.nnet(
                noisy_latents,
                timesteps,
                text_embedding,
                vector_embedding,
                # added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=None,  # [sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=None,  # mid_block_res_sample.to(dtype=weight_dtype),
                controls=controls,
            )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps, batch)
        return loss
