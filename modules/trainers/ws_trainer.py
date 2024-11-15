import torch
import gc
import clip
import random
import numpy as np
from typing import Union
from .base_trainer import BaseTrainer
from ..models.ws.mlp import MLP4
from ..utils import ws_train_utils
from ..datasets.ws_dataset import WaifuScorerDataset
from ..train_state.ws_train_state import WaifuScorerTrainState


class WaifuScorerTrainer(BaseTrainer):
    pretrained_model_name_or_path: Union[str, None] = None
    clip_model_name_or_path: str = "ViT-L/14"

    input_size: int = 768

    cache_column: str = "cache"

    mlp: MLP4
    mlp_class = MLP4
    dataset_class = WaifuScorerDataset
    train_state_class = WaifuScorerTrainState

    cache_emb: bool = False
    cache_only: bool = False

    def get_setups(self):
        if self.cache_emb and self.cache_only:
            return [
                self._setup_dtype,
                self._setup_model,
                self._setup_dataset
            ]
        else:
            return super().get_setups()

    def get_model_loaders(self):
        if self.cache_only:
            return [self.load_clip_model]
        else:
            return super().get_model_loaders()

    def _setup_dataset(self):
        super()._setup_dataset()
        if self.cache_emb:
            self.cache_image_embeddings()

    def cache_image_embeddings(self):
        with torch.no_grad():
            self.train_dataset.cache_all_image_embeddings(self.clip_model, self.clip_preprocess)
            if self.valid_dataset is not None:
                self.valid_dataset.cache_all_image_embeddings(self.clip_model, self.clip_preprocess)
        del self.clip_model, self.clip_preprocess
        self.logger.info(f"unloaded clip model and preprocess")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.accelerator.wait_for_everyone()

    def load_clip_model(self):
        clip_model, clip_preprocess = clip.load(self.clip_model_name_or_path, device=self.device)  # RN50x64
        return {"clip_model": clip_model, "clip_preprocess": clip_preprocess}

    def load_mlp_model(self):
        model = self.mlp_class(
            input_size=self.input_size
        )
        if self.pretrained_model_name_or_path is not None:
            self.logger.info(f"loading pretrained model from {self.pretrained_model_name_or_path}")
            state_dict = torch.load(self.pretrained_model_name_or_path, map_location=self.device)
            model.load_state_dict(state_dict)
        return {"mlp": model}

    def setup_mlp_params(self):
        training_models = [self.mlp]
        params_to_optimize = [{'params': self.mlp.parameters(), 'lr': self.learning_rate}]
        if self.gradient_checkpointing:
            if hasattr(self.mlp, 'enable_gradient_checkpointing'):
                self.mlp.enable_gradient_checkpointing()
            else:
                self.logger.warning(f"Gradient checkpointing is not supported for {self.mlp.__class__.__name__}")
        self.mlp.requires_grad_(True)
        self.mlp.to(self.device, dtype=self.weight_dtype)
        self.mlp = self._prepare_one_model(self.mlp, train=True, name='mlp', transform_model_if_ddp=True)
        return training_models, params_to_optimize

    # def get_train_state(self):
    #     return self.train_state_class.from_config(
    #         self.config,
    #         self,
    #         self.accelerator,
    #         train_dataset=self.train_dataset,
    #         valid_dataset=self.valid_dataset,
    #         optimizer=self.optimizer,
    #         lr_scheduler=self.lr_scheduler,
    #         train_dataloader=self.train_dataloader,
    #         valid_dataloader=self.valid_dataloader,
    #         save_dtype=self.save_dtype,
    #         mlp=self.mlp,
    #     )

    def train_step(self, batch):
        if (img_emb := batch.get("image_embeddings", None)) is not None:
            img_emb = img_emb.to(self.device)
        else:
            images = batch['images']
            with torch.no_grad():
                img_emb = ws_train_utils.encode_images(images, self.clip_model, self.clip_preprocess, device=self.device, max_workers=self.train_dataset.max_dataset_n_workers)
        img_emb.to(self.device)

        with self.accelerator.autocast():
            output = self.mlp(img_emb)
            if torch.isnan(output).any():
                raise ValueError(f"output is NaN at step {self.train_state.global_step}")

        scores = batch['scores'].to(self.device)
        # loss = torch.mean(torch.abs(output - scores)) #mae
        loss = torch.mean((output - scores)**2)  # mse

        if torch.isnan(loss):
            raise ValueError(f"loss is NaN at step {self.train_state.global_step}")
        return loss
