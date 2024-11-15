import torch
import torch.nn.functional as F
import lpips
from typing import Literal, Callable
from diffusers.training_utils import EMAModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from waifuset import logging
from .base_trainer import BaseTrainer
from ..train_state.vae_train_state import VAETrainState
from ..datasets.t2i_dataset import T2IDataset
from ..utils import sd15_model_utils, vae_train_utils


class VAETrainer(BaseTrainer):
    vae_model_name_or_path: str
    ema_vae_model_name_or_path: str = None
    lpips_model_name_or_path: str = None  # "auto"
    vae: AutoencoderKL

    diffusion_backbone: Literal['sd15', 'sdxl'] = 'sdxl'
    dataset_class = T2IDataset
    train_state_class = VAETrainState

    use_xformers: bool = True

    train_decoder_only: bool = True
    recon_loss_type: Literal['l1', 'l2', 'huber'] = 'l1'
    lpips_scale: float = 5e-1
    kl_scale: float = 1e-6
    patch_loss: bool = True
    patch_size: int = 64
    patch_stride: int = 32

    recon_loss_fn: Callable

    def load_vae_model(self):
        self.logger.info(f"Loading VAE model from {logging.yellow(self.vae_model_name_or_path)}")
        vae = sd15_model_utils.load_vae(self.vae_model_name_or_path, dtype=self.weight_dtype)
        return {"vae": vae}

    def load_lpips_loss_fn_model(self):
        self.logger.info(f"Loading LPIPS model from {logging.yellow(self.lpips_model_name_or_path)}")
        lpips_loss_fn = lpips.LPIPS(
            net="alex",
            model_path=self.lpips_model_name_or_path,
        )
        return {"lpips_loss_fn": lpips_loss_fn}

    def setup_ema(self):
        self.ema_models = {}

        self.ema_vae = EMAModel(self.vae.parameters(), model_cls=AutoencoderKL, model_config=self.vae.config)
        if self.ema_vae_model_name_or_path:
            self.logger.info(f"Loading EMA VAE model from {logging.yellow(self.ema_vae_model_name_or_path)}")
            self.ema_vae.load_state_dict(EMAModel.from_pretrained(self.vae_model_name_or_path, model_cls=AutoencoderKL).state_dict())
        else:
            self.logger.info("Creating EMA VAE model from scratch")
        self.ema_vae.to(self.device, dtype=self.weight_dtype)
        self.ema_vae = self._prepare_one_model(self.ema_vae, train=False, name='vae', transform_model_if_ddp=True)

        self.ema_models["vae"] = self.ema_vae

    def ema_step(self):
        self.ema_vae.step(self.vae.parameters())

    def get_vae_scale_factor(self):
        if self.diffusion_backbone == 'sd15':
            return 0.18215
        elif self.diffusion_backbone == 'sdxl':
            return 0.13025
        else:
            raise ValueError(f"Invalid backbone: {self.diffusion_backbone}")

    def setup_vae_params(self):
        self.vae.to(device=self.device, dtype=self.weight_dtype)
        self.vae.encoder.to(device=self.device, dtype=self.weight_dtype)
        self.vae.decoder.to(device=self.device, dtype=self.weight_dtype)

        if self.gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
        if self.use_xformers:
            if torch.__version__ >= "2.0.0":
                self.vae.set_use_memory_efficient_attention_xformers(True)
            else:
                self.logger.warning(f"XFormers not supported for vae in this PyTorch version. 2.0.0+ required, but got: {torch.__version__}")

        if self.train_decoder_only:
            self.vae.requires_grad_(True)
            self.vae.decoder.requires_grad_(True)
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
        else:
            self.vae.requires_grad_(True)
            self.vae.encoder.requires_grad_(True)
            self.vae.decoder.requires_grad_(True)
        self.vae.train()
        self.vae = self._prepare_one_model(self.vae, train=True, name='vae', transform_model_if_ddp=True)
        params_to_optimize = [{"params": list(self.vae.parameters()), "lr": self.learning_rate}]
        return [self.vae], params_to_optimize

    def setup_lpips_loss_fn_params(self):
        self.lpips_loss_fn.to(self.device, dtype=self.weight_dtype)
        self.lpips_loss_fn.requires_grad_(False)
        self.lpips_loss_fn.eval()
        self.lpips_loss_fn = self._prepare_one_model(self.lpips_loss_fn, train=False, name='lpips_loss_fn', transform_model_if_ddp=True)
        return [], []

    def _setup_training(self):
        super()._setup_training()
        self.vae_scale_factor = self.get_vae_scale_factor()
        self.logger.info(f"VAE scale factor: {self.vae_scale_factor}")

        if self.recon_loss_type.lower() in ('l1', 'mae'):
            self.recon_loss_fn = F.l1_loss
        elif self.recon_loss_type.lower() in ('l2', 'mse'):
            self.recon_loss_fn = F.mse_loss
        elif self.recon_loss_type.lower() in ('huber', 'smoothl1'):
            self.recon_loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Invalid recon loss type: {self.recon_loss_type}")

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
    #         vae=self.vae,
    #         ema_models=self.ema_models,
    #     )

    def get_start_training_message(self):
        msgs = super().get_start_training_message()
        if self.use_ema:
            msgs.append("  using EMA")
        if self.train_decoder_only:
            msgs.append("  train decoder only")
        else:
            msgs.append("  train full VAE")
        msgs.append(f"  reconstruction loss type: {self.recon_loss_type}")
        return msgs

    def train_step(self, batch):
        images = batch["images"].to(self.device, dtype=self.weight_dtype)
        if self.accelerator.num_processes > 1:
            posterior = self.vae.module.encode(images).latent_dist
            z = posterior.sample().to(self.weight_dtype)
            pred = self.vae.module.decode(z).sample.to(self.weight_dtype)
        else:
            posterior = self.vae.encode(images).latent_dist  # .to(weight_dtype)
            # z = mean                      if posterior.mode()
            # z = mean + variable*epsilon   if posterior.sample()
            z = posterior.sample().to(self.weight_dtype)  # Not mode()
            pred = self.vae.decode(z).sample.to(self.weight_dtype)

        # pred = pred#.to(dtype=weight_dtype)
        if not self.train_decoder_only:
            kl_loss = posterior.kl().mean().to(self.weight_dtype)
            kl_loss *= self.kl_scale

        if self.patch_loss:
            # patched loss
            recon_loss = vae_train_utils.patch_based_mse_loss(images, pred, self.patch_size, self.patch_stride).to(self.weight_dtype)
            lpips_loss = vae_train_utils.patch_based_lpips_loss(self.lpips_loss_fn, images, pred, self.patch_size, self.patch_stride).to(self.weight_dtype)

        else:
            # default loss
            recon_loss = self.recon_loss_fn(pred, images, reduction="mean").to(self.weight_dtype)
            with torch.no_grad():
                lpips_loss = self.lpips_loss_fn(pred, images).mean().to(self.weight_dtype)
                if not torch.isfinite(lpips_loss):
                    lpips_loss = torch.tensor(0)
                    self.logger.warning("LPIPS loss is not finite. Setting to 0.")

        lpips_loss *= self.lpips_scale

        if self.train_decoder_only:
            # remove kl term from loss, bc when we only train the decoder, the latent is untouched
            # and the kl loss describes the distribution of the latent
            loss = recon_loss + lpips_loss  # .to(weight_dtype)
        else:
            loss = recon_loss + lpips_loss + kl_loss  # .to(weight_dtype)

        if not torch.isfinite(loss):
            pred_mean = pred.mean()
            target_mean = images.mean()
            raise ValueError(f"Infinite loss incurred. pred_mean: {pred_mean}, target_mean: {target_mean}")

        if not self.train_decoder_only:
            self.pbar_logs["kl_loss"] = kl_loss.item()
        self.pbar_logs.update({"lpips_loss": lpips_loss.item(), "recon_loss": recon_loss.item()})

        return loss
