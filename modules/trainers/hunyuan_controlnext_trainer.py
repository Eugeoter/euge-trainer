import torch
from .hunyuan_trainer import HunyuanTrainer
from .sd15_controlnext_trainer import SD15ControlNeXtTrainer
from ..models.hunyuan.modules.models_controlnext import HunYuanDiT as HunYuanControlNeXtDiT
from ..models.hunyuan.modules.controlnext import ControlNetModel
# from ..pipelines.hunyuan_controlnext_pipeline import HunyuanDiTControlNeXtPipeline
from ..models.hunyuan.diffusion.pipeline_controlnext import StableDiffusionControlNeXtPipeline as HunyuanDiTControlNeXtPipeline
from ..train_state.hunyuan_controlnext_train_state import HunyuanControlNeXtTrainState
from ..datasets.hunyuan_controlnet_dataset import HunyuanControlNetDataset


class HunyuanControlNeXtTrainer(HunyuanTrainer, SD15ControlNeXtTrainer):
    nnet_class = HunYuanControlNeXtDiT
    pipeline_class = HunyuanDiTControlNeXtPipeline
    train_state_class = HunyuanControlNeXtTrainState
    dataset_class = HunyuanControlNetDataset
    controlnext_class = ControlNetModel

    def enable_xformers(self):
        HunyuanTrainer.enable_xformers(self)
        SD15ControlNeXtTrainer.enable_xformers(self)

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
    #     )

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        with torch.set_grad_enabled(self.train_text_encoder):
            encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5 = self.encode_caption(
                batch["captions"],
            )
            if self.full_fp16:
                encoder_hidden_states = encoder_hidden_states.to(self.weight_dtype)
                encoder_hidden_states_t5 = encoder_hidden_states_t5.to(self.weight_dtype)

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(self.weight_dtype)

        # B, C, H, W = noisy_latents.shape
        # freqs_cis_img = hunyuan_train_utils.calc_rope(H * 8, W * 8, 2, 88 if self.model_type == "DiT-g/2" else 72)
        # cos_cis_img, sin_cis_img = freqs_cis_img
        # self.logger.print(f"cos_cis_img.shape: {cos_cis_img.shape}, sin_cis_img.shape: {sin_cis_img.shape}")
        # self.logger.print(f"batch['cos_cis_img'].shape: {batch['cos_cis_img'].shape}, batch['sin_cis_img'].shape: {batch['sin_cis_img'].shape}")

        control_images = batch['control_images'].to(self.device, dtype=latents.dtype)

        # self.logger.debug(
        #     f"latents.shape: {latents.shape}, encoder_hidden_states.shape: {encoder_hidden_states.shape}, text_embedding_mask.shape: {text_embedding_mask.shape}, encoder_hidden_states_t5.shape: {encoder_hidden_states_t5.shape}, text_embedding_mask_t5.shape: {text_embedding_mask_t5.shape}, image_meta_size.shape: {batch['image_meta_size'].shape}, style.shape: {batch['style'].shape}, cos_cis_img.shape: {batch['cos_cis_img'].shape}, sin_cis_img.shape: {batch['sin_cis_img'].shape}")

        # check all the `requires_grad` flags
        # self.logger.debug(f"latents.requires_grad: {latents.requires_grad}")
        # self.logger.debug(f"encoder_hidden_states.requires_grad: {encoder_hidden_states.requires_grad}")
        # self.logger.debug(f"text_embedding_mask.requires_grad: {text_embedding_mask.requires_grad}")
        # self.logger.debug(f"encoder_hidden_states_t5.requires_grad: {encoder_hidden_states_t5.requires_grad}")
        # self.logger.debug(f"text_embedding_mask_t5.requires_grad: {text_embedding_mask_t5.requires_grad}")
        # self.logger.debug(f"noisy_latents.requires_grad: {noisy_latents.requires_grad}")
        # self.logger.debug(f"timesteps.requires_grad: {timesteps.requires_grad}")

        with self.accelerator.autocast():
            controls = self.controlnext(control_images, timesteps, return_dict=False)
            model_kwargs = {
                "encoder_hidden_states": encoder_hidden_states,
                "text_embedding_mask": text_embedding_mask,
                "encoder_hidden_states_t5": encoder_hidden_states_t5,
                "text_embedding_mask_t5": text_embedding_mask_t5,
                "image_meta_size": batch["image_meta_size"] if self.size_cond else None,
                "style": batch["style"] if self.use_style_cond else None,
                # "cos_cis_img": freqs_cis_img[0],
                # "sin_cis_img": freqs_cis_img[1],
                "cos_cis_img": batch['cos_cis_img'],
                "sin_cis_img": batch['sin_cis_img'],
                "controls": controls,
                "return_dict": False,
            }
            # self.logger.debug(f"x_t: {noisy_latents}\nt: {timesteps}\nmodel_kwargs: {model_kwargs}")
            # self.logger.debug(f"x_t.shape: {noisy_latents.shape}\nt.shape: {timesteps.shape}")
            noise_pred = self.nnet(
                noisy_latents,
                timesteps,
                **model_kwargs,
            )
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(noise_pred, target, timesteps, batch)

        return loss
