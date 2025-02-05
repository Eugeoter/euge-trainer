import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import einops
from typing import Literal, Callable, Dict, Union
from diffusers.training_utils import EMAModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from waifuset import logging
from .base_trainer import BaseTrainer
from ..train_state.vae_train_state import VAETrainState
from ..datasets.t2i_dataset import T2IDataset
from ..utils import sd15_model_utils, vae_train_utils, train_utils


class VAETrainer(BaseTrainer):
    vae_model_name_or_path: str
    ema_vae_model_name_or_path: str = None
    vae: AutoencoderKL

    diffusion_backbone: Literal['sd15', 'sdxl'] = 'sdxl'
    dataset_class = T2IDataset
    train_state_class = VAETrainState

    use_xformers: bool = True

    train_decoder_only: bool = True

    use_recon_loss: bool = True
    lambda_recon: float = 1.0
    recon_loss_fn: Callable
    recon_loss_type: Literal['l1', 'l2', 'huber'] = 'l1'

    use_lpips_loss: bool = True
    lambda_lpips: float = 1e-5
    lpips_model_name_or_path: str = None  # "auto"
    lpips_model_type: Literal['vgg19', 'alex'] = 'alex'

    use_kld_loss: bool = False
    lambda_kld: float = 1.0
    kld_patch_size: int = 32

    use_gan_loss: bool = True
    lambda_gan: float = 1e-2
    gan_disc_type: str = "bce"
    lr_discriminator: float = 1e-3
    disc_optimizer_type: str = "Adafactor"
    disc_optimizer_kwargs: Dict[str, Union[str, float]] = dict(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    disc_lr_scheduler_type: str = "constant_with_warmup"
    disc_lr_warmup_steps: int = 0
    disc_lr_scheduler_num_cycles: int = 1
    disc_lr_scheduler_power: float = 1.0
    disc_lr_scheduler_kwargs: Dict[str, Union[str, float]] = {}

    def load_vae_model(self):
        self.logger.info(f"Loading VAE model from {logging.yellow(self.vae_model_name_or_path)}")
        vae = sd15_model_utils.load_vae(self.vae_model_name_or_path, dtype=self.weight_dtype)
        return {"vae": vae}

    def load_lpips_loss_fn_model(self):
        self.logger.info(f"Loading LPIPS model from {logging.yellow(self.lpips_model_name_or_path)}")
        lpips_loss_fn = lpips.LPIPS(
            net=self.lpips_model_type,
            model_path=self.lpips_model_name_or_path,
        )
        return {"lpips_loss_fn": lpips_loss_fn}

    def load_discriminator_model(self):
        from ..models.gan import gan
        discriminator = gan.NLayerDiscriminator(input_nc=3).to(self.device)
        discriminator.apply(gan.weights_init)
        return {'discriminator': discriminator}

    def _setup_optims(self):
        super()._setup_optims()

        if self.use_gan_loss:
            self.logger.info("Setting up discriminator optimizer and LR scheduler")
            self.disc_optimizer = train_utils.get_optimizer(
                optimizer_type=self.disc_optimizer_type,
                trainable_params=self.discriminator.parameters(),
                lr=self.lr_discriminator,
                lr_scheduler_type=self.disc_lr_scheduler_type,
                **self.disc_optimizer_kwargs,
            )
            self.disc_lr_scheduler = train_utils.get_scheduler_fix(
                lr_scheduler_type=self.disc_lr_scheduler_type,
                optimizer=self.disc_optimizer,
                num_train_steps=self.num_train_steps,
                num_warmup_steps=self.disc_lr_warmup_steps,
                num_cycles=self.disc_lr_scheduler_num_cycles,
                power=self.disc_lr_scheduler_power,
                **self.disc_lr_scheduler_kwargs,
            )
            self.disc_optimizer = self.accelerator.prepare(self.disc_optimizer)
            self.disc_lr_scheduler = self.accelerator.prepare(self.disc_lr_scheduler)

    def setup_discriminator_params(self):
        self.discriminator.requires_grad_(True)
        self.discriminator.to(self.device, dtype=torch.float32)
        self.discriminator.train()
        self.discriminator = self._prepare_one_model(self.discriminator, train=True, name="discriminator", transform_model_if_ddp=True)
        return [], []  # manually optimize

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

    def is_disc_step(self):
        return self.use_gan_loss and self.train_state.global_step % 2 == 0

    def is_gen_step(self):
        return self.use_gan_loss and self.train_state.global_step % 2 == 1

    def patch_nce(self, p_psi, p_theta, p_psi_tau, p_phi, patch_num=4):
        # 这里是随机选取patch_num个patch，计算patch之间的相似度
        # 正样本
        # p_psi 要和 p_theta 对应的patch 相似
        # p_psi_tau 要和 p_phi 对应的patch 相似
        # 负样本
        # p_psi 要和 p_phi 对应的patch 不相似
        # 输入的格式   预测结果: (b, c, h, w)
        # (b, c, h, w) -> (b, c, h*w)

        total_loss = 0
        p_psi = p_psi.view(p_psi.shape[0], p_psi.shape[1], -1)
        p_theta = p_theta.view(p_theta.shape[0], p_theta.shape[1], -1)
        p_psi_tau = p_psi_tau.view(p_psi_tau.shape[0], p_psi_tau.shape[1], -1)
        p_phi = p_phi.view(p_phi.shape[0], p_phi.shape[1], -1)
        # 随机选取patch_num个patch
        p_shape = p_psi.shape
        patch_num = min(patch_num, p_shape[2])
        patch_index = torch.randperm(p_shape[2])[:patch_num]
        # 从 p_psi, p_theta, p_psi_tau, p_phi 中提取 对应的patch 并且将 (b,c,p) 展平 为 (b*p, c)
        p_psi_patch = p_psi[:, :, patch_index].permute(0, 2, 1).reshape(-1, p_psi.shape[1])
        p_theta_patch = p_theta[:, :, patch_index].permute(0, 2, 1).reshape(-1, p_theta.shape[1])
        p_psi_tau_patch = p_psi_tau[:, :, patch_index].permute(0, 2, 1).reshape(-1, p_psi_tau.shape[1])
        p_phi_patch = p_phi[:, :, patch_index].permute(0, 2, 1).reshape(-1, p_phi.shape[1])
        # 先都归一化
        p_psi_patch = torch.nn.functional.normalize(p_psi_patch, p=2, dim=1)
        p_theta_patch = torch.nn.functional.normalize(p_theta_patch, p=2, dim=1)
        p_psi_tau_patch = torch.nn.functional.normalize(p_psi_tau_patch, p=2, dim=1)
        p_phi_patch = torch.nn.functional.normalize(p_phi_patch, p=2, dim=1)
        # 计算相似度
        # 正样本
        loss_positive = self.patch_loss(p_psi_patch, p_theta_patch) + self.patch_loss(p_psi_tau_patch, p_phi_patch)
        # 负样本
        # loss_negative = self.patchloss(p_psi_patch,p_psi_tau_patch)
        total_loss = loss_positive.mean()
        # total_loss = loss_positive.mean() + loss_negative.mean()
        # self.logger.debug(f"loss_positive: {loss_positive.mean().item()} | loss_negative: {loss_negative.mean().item()} | total_loss: {total_loss.mean().item()}")
        return total_loss.mean()

    def patch_loss(self, feature1, feature2, batch_size=1, nce_T=0.07):
        # print(f"feature1 dtype: {feature1.dtype}, feature2 dtype: {feature2.dtype}")
        # 全部转换为float32
        feature1 = feature1.to(dtype=torch.float32)
        feature2 = feature2.to(dtype=torch.float32)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = feature1.shape[0]
        dim = feature1.shape[1]
        # feature2 = feature2.detach()  # 阻止梯度传播到键特征

        # 计算正样本的相似度（对角线元素）
        l_pos = torch.bmm(
            feature1.view(num_patches, 1, -1),
            feature2.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # 计算负样本的相似度
        feat_q = feature1.view(batch_size, -1, dim)
        feature2 = feature2.view(batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feature2.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # 拼接正负样本并应用温度缩放
        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        # 标签 0 标识正样本
        labels = torch.zeros(out.size(0), dtype=torch.long, device=feature1.device)
        loss = cross_entropy_loss(out, labels)

        return loss

    def kl_divergence(self, reconstructed_tensor, groundtruth_tensor, path_size=32, overlap=True, weight=1000):
        B, C, H, W = reconstructed_tensor.shape
        # print(f'H: {H}, W: {W}')
        # self.logger.debug(f'H: {H}, W: {W}')
        # self.logger.debug(f'path_size: {path_size}')
        assert H % path_size == 0 and W % path_size == 0

        def kld_loss(reconstructed, groundtruth, patch=True):
            dtype = reconstructed.dtype
            reconstructed = reconstructed.to(dtype=torch.float32)
            groundtruth = groundtruth.to(dtype=torch.float32)
            if patch:
                reconstructed = einops.rearrange(reconstructed, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)
                groundtruth = einops.rearrange(groundtruth, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)

                p = F.softmax(groundtruth, dim=-1)
                q = F.softmax(reconstructed, dim=-1)
                log_p = F.log_softmax(groundtruth, dim=-1)
                log_q = F.log_softmax(reconstructed, dim=-1)
                kl_div = torch.sum(p * (log_p - log_q), dim=-1) * weight
                # kl_div = torch.sum(torch.abs(p * log_p - q * log_q), dim=-1) * weight
                # kl_div = torch.sum(torch.abs(p - q), dim=-1) * weight
            else:
                p = F.softmax(groundtruth, dim=1)
                q = F.softmax(reconstructed, dim=1)
                log_p = F.log_softmax(groundtruth, dim=1)
                log_q = F.log_softmax(reconstructed, dim=1)
                kl_div = torch.sum(p * (log_p - log_q), dim=1) * weight
                # kl_div = torch.sum(torch.abs(p * log_p - q * log_q), dim=1) * weight
                # kl_div = torch.sum(torch.abs(p  - q ), dim=1) * weight
            return kl_div.mean().to(dtype=dtype)

        loss = kld_loss(reconstructed_tensor, groundtruth_tensor)
        if overlap:
            shift_size = path_size // 2
            shifted_H_reconstructed = torch.zeros_like(reconstructed_tensor)
            shifted_H_groundtruth = torch.zeros_like(groundtruth_tensor)
            shifted_H_reconstructed[:, :, shift_size:, :] = reconstructed_tensor[:, :, :-shift_size, :]
            shifted_H_reconstructed[:, :, :shift_size, :] = reconstructed_tensor[:, :, -shift_size:, :]
            shifted_H_groundtruth[:, :, shift_size:, :] = groundtruth_tensor[:, :, :-shift_size, :]
            shifted_H_groundtruth[:, :, :shift_size, :] = groundtruth_tensor[:, :, -shift_size:, :]

            shifted_HW_reconstructed = torch.zeros_like(shifted_H_reconstructed)
            shifted_HW_groundtruth = torch.zeros_like(shifted_H_groundtruth)
            shifted_HW_reconstructed[:, :, :, shift_size:] = shifted_H_reconstructed[:, :, :, :-shift_size]
            shifted_HW_reconstructed[:, :, :, :shift_size] = shifted_H_reconstructed[:, :, :, -shift_size:]
            shifted_HW_groundtruth[:, :, :, shift_size:] = shifted_H_groundtruth[:, :, :, :-shift_size]
            shifted_HW_groundtruth[:, :, :, :shift_size] = shifted_H_groundtruth[:, :, :, -shift_size:]

            loss += kld_loss(shifted_HW_reconstructed, shifted_HW_groundtruth)
            loss /= 2

        loss += kld_loss(reconstructed_tensor, groundtruth_tensor, patch=False)

        return loss

    def get_loss(self, model_pred, target, **kwargs):
        model_pred = model_pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        # GAN (Discriminator) loss
        if self.is_disc_step():
            logits_real = self.discriminator(target.detach())
            logits_fake = self.discriminator(model_pred.detach())
            d_loss = vae_train_utils.hinge_d_loss(logits_real, logits_fake) * self.lambda_gan
            loss = d_loss

            # self.logger.debug(f"Discriminator loss: {d_loss.item()}")
            self.accelerator_logs.update({
                f"disc_loss/step": d_loss.item(),
            })

        else:
            loss = 0
            # Reconstruction loss
            if self.use_recon_loss and self.lambda_recon > 0:
                recon_loss = self.recon_loss_fn(model_pred, target) * self.lambda_recon
                loss += recon_loss

            # Perceptual loss
            if self.use_lpips_loss and self.lambda_lpips > 0:
                try:
                    if target.shape[1] < 3:
                        # We'll put zeros in the third channel...
                        n_to_add = 3-target.shape[1]
                        target_pad = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, n_to_add), mode='constant', value=0)
                        model_pred_pad = torch.nn.functional.pad(model_pred, (0, 0, 0, 0, 0, n_to_add), mode='constant', value=0)
                        lpips_loss_batch = self.lpips_loss_fn(model_pred_pad, target_pad).mean()
                    elif target.shape[1] > 3:
                        lpips_loss_batch = self.lpips_loss_fn(model_pred[:, :3, :, :], target[:, :3, :, :]).mean()
                    else:
                        lpips_loss_batch = self.lpips_loss_fn(model_pred, target).mean()
                except Exception as e:
                    self.logger.error(f"Error in calculating LPIPS loss: {e}. Model pred shape: {model_pred.shape}, target shape: {target.shape}")
                    lpips_loss_batch = torch.tensor(0.0, device=self.device, dtype=self.weight_dtype)
                lpips_loss_batch *= self.lambda_lpips
                loss += lpips_loss_batch

            # KL divergence loss
            if self.use_kld_loss and self.lambda_kld > 0:
                kld_loss = self.kl_divergence(model_pred, target, path_size=self.kld_patch_size, overlap=True, weight=10) * self.lambda_kld
                loss += kld_loss

            # GAN (Generator) loss
            if self.is_gen_step():
                logits_real = self.discriminator(target)
                logits_fake = self.discriminator(model_pred.to(dtype=torch.float32))
                g_loss = vae_train_utils.hinge_d_loss(logits_real, logits_fake) * self.lambda_gan
                loss += g_loss

            self.accelerator_logs.update({
                f"recon_loss/step": recon_loss.item(),
                f"kld_loss/step": kld_loss.item() if self.is_gen_step() else None,
                f"g_loss/step": g_loss.item() if self.is_gen_step() else None,
                f"lpips_loss/step": lpips_loss_batch.item() if self.lambda_lpips else None,
            })

        return loss

    def optimizer_step(self, loss):
        if self.is_disc_step():
            self.disc_optimizer.step()
        else:
            super().optimizer_step(loss)

    def lr_scheduler_step(self):
        if self.is_disc_step():
            self.disc_lr_scheduler.step()
        else:
            super().lr_scheduler_step()

    def zero_grad(self):
        if self.is_disc_step():
            self.disc_optimizer.zero_grad(set_to_none=True)
        else:
            super().zero_grad()
            # if self.is_gen_step():
            #     self.disc_optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, **kwargs):
        return super().backward(loss, **kwargs)

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

        loss = self.get_loss(pred, images)

        return loss
