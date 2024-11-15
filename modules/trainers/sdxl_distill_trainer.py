import copy
import torch
from .sd_distill_trainer import DistillTrainer
from .sdxl_trainer import SDXLTrainer
from ..utils import sdxl_train_utils
from ..datasets.sdxl_distill_dataset import SDXLDistillDataset


class SDXLDistillTrainer(SDXLTrainer, DistillTrainer):
    dataset_class = SDXLDistillDataset

    def setup(self):
        self.check_config()
        if self.cache_latents and self.cache_only:
            setups = [
                self._setup_basic,
                self._setup_dtype,
                self._setup_model,
                self._setup_dataset
            ]
        else:
            setups = [
                self._setup_basic,
                self._setup_dtype,
                self._setup_model,
                self._setup_dataset,
                self._setup_training,
                self._setup_params,
                self._setup_teacher_models,
                self._setup_optims,
                self._setup_diffusion,
                self._setup_loss_recorder,
                self._setup_train_state,
            ]
        for setup in setups:
            setup()

    def setup_teacher_text_encoder(self):
        if self.train_text_encoder1:
            teacher_text_encoder1 = copy.deepcopy(self.text_encoder1)
            teacher_text_encoder1.eval()
            teacher_text_encoder1.requires_grad_(False)
            for param in teacher_text_encoder1.parameters():
                param.requires_grad = False
        else:
            teacher_text_encoder1 = self.text_encoder1
        teacher_text_encoder1.to(self.text_encoder1.device)
        self.teacher_text_encoder1 = teacher_text_encoder1
        return teacher_text_encoder1

    def setup_teacher_text_encoder2(self):
        if self.train_text_encoder2:
            teacher_text_encoder2 = copy.deepcopy(self.text_encoder2)
            teacher_text_encoder2.eval()
            teacher_text_encoder2.requires_grad_(False)
            for param in teacher_text_encoder2.parameters():
                param.requires_grad = False
        else:
            teacher_text_encoder2 = self.text_encoder2
        teacher_text_encoder2.to(self.text_encoder2.device)
        self.teacher_text_encoder2 = teacher_text_encoder2
        return teacher_text_encoder2

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
        student_text_embedding, student_vector_embedding = self.get_embeddings(
            batch['captions'],
            target_size,
            orig_size,
            crop_size,
            batch['negative_captions'] if self.do_classifier_free_guidance else None
        )
        teacher_text_embedding, teacher_vector_embedding = self.get_teacher_embeddings(
            batch['teacher_captions'],
            target_size,
            orig_size,
            crop_size,
            batch['teacher_negative_captions'] if self.do_classifier_free_guidance else None
        )
        student_text_embedding = student_text_embedding.to(self.weight_dtype)
        student_vector_embedding = student_vector_embedding.to(self.weight_dtype)
        teacher_text_embedding = teacher_text_embedding.to(self.weight_dtype)
        teacher_vector_embedding = teacher_vector_embedding.to(self.weight_dtype)

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(self.weight_dtype)

        with self.accelerator.autocast():
            student_pred = self.nnet(noisy_latents, timesteps, student_text_embedding, student_vector_embedding)
            with torch.no_grad():
                teacher_pred = self.teacher_nnet(noisy_latents, timesteps, teacher_text_embedding, teacher_vector_embedding)

        loss = self.get_loss(student_pred, teacher_pred)
        return loss

    def get_loss(self, student_pred, teacher_pred):
        return torch.nn.functional.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")

    def get_teacher_embeddings(self, captions, target_size, orig_size, crop_size, negative_captions=None):
        text_embeddings1, text_embeddings2, text_pool2, uncond_embeddings1, uncond_embeddings2, uncond_pool2 = self.encode_teacher_caption(captions, negative_captions)
        size_embeddings = sdxl_train_utils.get_size_embeddings(orig_size, crop_size, target_size, self.device)

        if self.do_classifier_free_guidance:
            text_embeddings = torch.cat([text_embeddings1, text_embeddings2], dim=2)
            uncond_embeddings = torch.cat([uncond_embeddings1, uncond_embeddings2], dim=2)
            text_embedding = torch.cat([uncond_embeddings, text_embeddings])

            cond_vector = torch.cat([text_pool2, size_embeddings], dim=1)
            uncond_vector = torch.cat([uncond_pool2, size_embeddings], dim=1)
            vector_embedding = torch.cat([uncond_vector, cond_vector])
        else:
            text_embedding = torch.cat([text_embeddings1, text_embeddings2], dim=2)
            vector_embedding = torch.cat([text_pool2, size_embeddings], dim=1)

        return text_embedding, vector_embedding

    def encode_teacher_caption(
        self,
        captions,
        negative_captions=None,
    ):
        with torch.no_grad():
            text_embeddings1, text_pool1, uncond_embeddings1, uncond_pool1 = sdxl_train_utils.encode_caption(
                self.teacher_text_encoder1,
                self.tokenizer1,
                captions,
                negative_captions=negative_captions,
                max_embeddings_multiples=self.max_embeddings_multiples,
                clip_skip=self.clip_skip,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                is_sdxl_text_encoder2=False,
            )
        with torch.no_grad():
            text_embeddings2, text_pool2, uncond_embeddings2, uncond_pool2 = sdxl_train_utils.encode_caption(
                self.teacher_text_encoder2,
                self.tokenizer2,
                captions,
                negative_captions=negative_captions,
                max_embeddings_multiples=self.max_embeddings_multiples,
                clip_skip=self.clip_skip,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                is_sdxl_text_encoder2=True,
            )
        return text_embeddings1, text_embeddings2, text_pool2, uncond_embeddings1, uncond_embeddings2, uncond_pool2
