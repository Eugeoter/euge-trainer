import torch
from .sd15_controlnext_trainer import SD15ControlNeXtTrainer
from .flux_trainer import FluxTrainer
from ..models.flux.flux_controlnext_models import FluxWithControlNeXt
from ..models.flux.flux_controlnext import ControlNetModel
from ..pipelines.flux_controlnext_pipeline import FluxControlNeXtPipeline
from ..train_state.flux_controlnext_train_state import FluxControlNeXtTrainState
from ..datasets.controlnet_dataset import ControlNetDataset
from ..utils import flux_train_utils, flux_model_utils


class FluxControlNeXtTrainer(SD15ControlNeXtTrainer, FluxTrainer):
    nnet_class = FluxWithControlNeXt
    pipeline_class = FluxControlNeXtPipeline
    train_state_class = FluxControlNeXtTrainState
    dataset_class = ControlNetDataset
    controlnext_class = ControlNetModel

    # def get_train_state(self):
    #     return self.train_state_class.from_config(
    #         self.config,
    #         self,
    #         self.accelerator,
    #         train_dataset=self.train_dataset,
    #         pipeline_class=self.pipeline_class,
    #         optimizer=self.optimizer,
    #         lr_scheduler=self.lr_scheduler,
    #         train_dataloader=self.train_dataloader,
    #         valid_dataloader=self.valid_dataloader,
    #         save_dtype=self.save_dtype,
    #         nnet=self.nnet,
    #         text_encoder=[self.text_encoder1, self.text_encoder2],
    #         tokenizer=[self.tokenizer1, self.tokenizer2],
    #         noise_scheduler=self.noise_scheduler,
    #         vae=self.vae,
    #         train_nnet=self.train_nnet,
    #         train_text_encoder=[self.train_text_encoder1, self.train_text_encoder2],
    #         controlnext=self.controlnext,
    #     )

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device, dtype=self.weight_dtype)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.device, dtype=self.vae.dtype))

        text_encoder_conds = self.encode_caption(batch['captions'])

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            self, self.noise_scheduler, latents, noise, self.accelerator.device, self.weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_model_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_model_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=self.accelerator.device)

        # get guidance
        guidance_vec = torch.full((bsz,), float(self.guidance_scale), device=self.accelerator.device)

        # call model
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not self.apply_t5_attn_mask:
            t5_attn_mask = None

        control_images = batch['control_images'].to(self.device, dtype=self.controlnext.dtype)
        controls = self.controlnext(control_images, timesteps)
        controls['scale'] = controls['scale'] * self.control_scale

        with self.accelerator.autocast():
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
            model_pred = self.nnet(
                img=packed_noisy_model_input,
                img_ids=img_ids,
                txt=t5_out,
                txt_ids=txt_ids,
                y=l_pooled,
                timesteps=timesteps / 1000,
                guidance=guidance_vec,
                txt_attention_mask=t5_attn_mask,
                controls=controls,
            )

        # unpack latents
        model_pred = flux_model_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(self, model_pred, noisy_model_input, sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        loss = self.get_loss(model_pred, target, timesteps, batch)
        if weighting is not None:
            loss = loss * weighting
        # if self.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
        #     loss = advanced_train_utils.apply_masked_loss(loss, batch)
        loss = loss.mean()
        return loss
