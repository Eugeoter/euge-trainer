import torch
import copy
from waifuset import logging
from .sd15_trainer import SD15Trainer

logging.warning(f"DistillTrainer is still in development, please use with caution!")


class DistillTrainer(SD15Trainer):
    def _setup_teacher_models(self):
        self.logger.info("setup teacher models...")
        for teacher_model_setter in dir(self):
            if teacher_model_setter.startswith('setup_teacher_') and callable(getattr(self, teacher_model_setter)):
                teacher_model = getattr(self, teacher_model_setter)()
                self.logger.info(f"  setup teacher model: {teacher_model.__class__.__name__}")

    def setup_teacher_nnet(self):
        if self.train_nnet:
            teacher_nnet = copy.deepcopy(self.nnet)
            teacher_nnet.eval()
            teacher_nnet.requires_grad_(False)
            for param in teacher_nnet.parameters():
                param.requires_grad = False
        else:
            teacher_nnet = self.nnet
        teacher_nnet.to(self.nnet.device)
        self.teacher_nnet = teacher_nnet
        return teacher_nnet

    def setup_teacher_text_encoder(self):
        if self.train_text_encoder:
            teacher_text_encoder = copy.deepcopy(self.text_encoder)
            teacher_text_encoder.eval()
            teacher_text_encoder.requires_grad_(False)
            for param in teacher_text_encoder.parameters():
                param.requires_grad = False
        else:
            teacher_text_encoder = self.text_encoder
        teacher_text_encoder.to(self.text_encoder.device)
        self.teacher_text_encoder = teacher_text_encoder
        return teacher_text_encoder

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        student_encoder_hidden_states = self.encode_caption(batch['captions'])
        teacher_encoder_hidden_states = self.encode_caption(batch['teacher_captions'])

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        with self.accelerator.autocast():
            student_pred = self.nnet(noisy_latents, timesteps, student_encoder_hidden_states).sample
            with torch.no_grad():
                teacher_pred = self.teacher_nnet(noisy_latents, timesteps, teacher_encoder_hidden_states).sample

        loss = self.get_loss(student_pred, teacher_pred)
        return loss

    def get_loss(self, student_pred, teacher_pred):
        return torch.nn.functional.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")
