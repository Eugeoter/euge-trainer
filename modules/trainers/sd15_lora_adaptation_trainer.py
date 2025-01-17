import torch
from einops import rearrange
from torch.nn import functional as F
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, List, Union, Literal
from safetensors.torch import load_file
from waifuset import logging
from .sd15_trainer import SD15Trainer
from ..utils import lora_utils, sd15_train_utils, vae_train_utils, train_utils
from ..models.sd15 import lora_adapter
from ..train_state.sd15_lora_adaptation_train_state import SD15LoRAAdaptationTrainState


class SD15LoRAAdaptationTrainer(SD15Trainer):
    backbone_type: str = 'sd15'

    # (i) path to a single lora model, (ii) list of paths to multiple lora models, or (iii) dictionary mapping model path to lora trigger word
    pretrained_lora_model_name_or_path: Union[str, Dict[str, str], List[str], List[Dict[str, str]]] = None

    train_state_class = SD15LoRAAdaptationTrainState

    nnet_phi: torch.nn.Module
    text_encoder_phi: torch.nn.Module
    nnet_psi: torch.nn.Module
    text_encoder_psi: torch.nn.Module

    lora_adapter_class = lora_adapter.LayerWiseMultiLoRAAdapter
    lora_adapter_type: Literal['layerwise', 'elementwise'] = 'layerwise'
    lr_w0: float = None
    lr_w1: List[float] = None

    loss_beta_1: float = 1.0
    loss_beta_2: float = 1.0

    loras: List[Dict[str, torch.Tensor]]
    taus_phi: List[str]
    taus_psi: List[str]

    init_w0: float = None
    init_w1: List[float] = None
    lora_strength: List[float] = None

    lambda_lpips: float = 0

    use_gan: bool = False
    use_lecam: bool = False
    gan_disc_type: str = "bce"
    lambda_gan: float = 0.01
    lambda_kld: float = 1.0
    kld_patch_size: int = 4
    lecam_loss_weight = 0.1
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9
    lr_discriminator: float = 1e-3
    disc_optimizer_type: str = "AdamW"
    disc_optimizer_kwargs: Dict[str, Union[str, float]] = {}
    disc_lr_scheduler_type: str = "constant_with_warmup"
    disc_lr_warmup_steps: int = 0
    disc_lr_scheduler_num_cycles: int = 1
    disc_lr_scheduler_power: float = 1.0
    disc_lr_scheduler_kwargs: Dict[str, Union[str, float]] = {}

    def get_setups(self):
        return super().get_setups() + [self.setup_lpips]

    def get_model_loaders(self):
        loaders = super().get_model_loaders()
        tails = [self.load_lora_model, self.load_phi_model, self.load_psi_model]
        for tail in tails:
            loaders.remove(tail)
            loaders.append(tail)
        return loaders

    def load_lora_model(self):
        if isinstance(self.pretrained_lora_model_name_or_path, str):
            lora_dicts = [dict(path=self.pretrained_lora_model_name_or_path, trigger='', tau='')]
        elif isinstance(self.pretrained_lora_model_name_or_path, list):
            for lora_src in self.pretrained_lora_model_name_or_path:
                if isinstance(lora_src, str):
                    lora_dicts = [dict(path=lora_src, trigger='', tau='')]
                elif isinstance(lora_src, dict):
                    lora_dicts = [lora_src]
                else:
                    raise ValueError(f"Invalid type for pretrained_lora_model_name_or_path: {type(lora_src)}")
        elif isinstance(self.pretrained_lora_model_name_or_path, (dict, ConfigDict)):
            lora_dicts = [dict(self.pretrained_lora_model_name_or_path)]
        else:
            raise ValueError(f"Invalid type for pretrained_lora_model_name_or_path: {type(self.pretrained_lora_model_name_or_path)}")

        loras = []
        taus_phi = []
        taus_psi = []
        for i, lora_dict in enumerate(lora_dicts):
            lora_path = lora_dict['path']
            tau_phi = lora_dict.get('tau_phi', '')
            tau_psi = lora_dict.get('tau_psi', '')

            self.logger.info(f"Loading the {i}-th LoRA model from {logging.yellow(lora_path)}")
            lora_state_dict = load_file(lora_path)

            loras.append(lora_state_dict)
            taus_phi.append(tau_phi)
            taus_psi.append(tau_psi)

        self.lora_name_to_weight_shape = lora_utils.make_lora_name_to_weight_shape_map(loras, model_type=self.backbone_type)
        self.taus_phi = taus_phi
        self.taus_psi = taus_psi

        return {'loras': loras}

    # def get_lora_adapter_dim(self):
    #     return max(len(lora) for lora in self.loras) // 3  # up, down & alpha
        # return 350

    # def load_lora_adapter_model(self):
    #     # self.lora_adapter_dim = self.get_lora_adapter_dim()
    #     self.num_loras = len(self.loras)
    #     self.logger.info(f"  LoRA adapter class: {logging.yellow(self.lora_adapter_class)}")
    #     # self.logger.info(f"LoRA adapter dimension: {self.lora_adapter_dim}")
    #     self.logger.info(f"  LoRA adapter dimension (used): {logging.yellow(max(len(lora_sd) for lora_sd in self.loras) // 3)}")
    #     self.logger.info(f"  Number of LoRAs: {self.num_loras}")
    #     lora_adapter = self.lora_adapter_class(
    #         lora_name_to_weight_shape=self.lora_name_to_weight_shape,
    #         num_loras=self.num_loras,
    #     )
    #     return {'lora_adapter': lora_adapter}

    def load_phi_model(self):
        if not hasattr(self, 'nnet'):
            raise ValueError("nnet is not loaded yet, please load nnet first.")
        if not hasattr(self, 'text_encoder'):
            raise ValueError("text_encoder is not loaded yet, please load text_encoder first.")
        if not hasattr(self, 'loras'):
            raise ValueError("loras is not loaded yet, please load loras first.")

        if self.lora_strength is None:
            self.lora_strength = [1.0] * len(self.loras)
        elif isinstance(self.lora_strength, (int, float)):
            self.lora_strength = [self.lora_strength] * len(self.loras)
        elif isinstance(self.lora_strength, list):
            if len(self.lora_strength) != len(self.loras):
                raise ValueError(f"Length of lora_strength ({len(self.lora_strength)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for lora_strength: {type(self.lora_strength)}")

        self.logger.info(f"LoRA strength: {logging.yellow(self.lora_strength)}")

        models_phi = lora_utils.merge_loras_to_model(
            {
                'nnet': self.nnet,
                'text_encoder': self.text_encoder,
            },
            lora_state_dicts=self.loras,
            lora_strength=self.lora_strength,
            model_type=self.backbone_type,
            merge_device=self.device,
            merge_dtype=self.weight_dtype,
            # name_to_module=self.lora_name_to_module,
            inplace=False,
        )
        self.models_phi = {model_name + '_phi': model for model_name, model in models_phi.items()}

        return self.models_phi

    def load_psi_model(self):
        r"""
        Initialize psi models by wrapping lora to model. Merge weights are calculated by the initialized LoRA adapter.
        """
        if not hasattr(self, 'nnet'):
            raise ValueError("nnet is not loaded yet, please load nnet first.")
        if not hasattr(self, 'text_encoder'):
            raise ValueError("text_encoder is not loaded yet, please load text_encoder first.")

        if self.lora_adapter_type == 'layerwise':
            self.lora_adapter_class = lora_adapter.LayerWiseMultiLoRAAdapter
        elif self.lora_adapter_type == 'elementwise':
            self.lora_adapter_class = lora_adapter.ElementWiseMultiLoRAAdapter
        else:
            raise ValueError(f"Invalid lora_adapter_type: {self.lora_adapter_type}, expected 'layerwise' or 'elementwise'")

        self.lora_name_to_orig_module = lora_utils.make_lora_name_to_module_map([self.nnet, self.text_encoder], model_type=self.backbone_type, debug_te=False)
        self.lora_name_to_orig_module_name = lora_utils.make_lora_name_to_module_name_map([self.nnet, self.text_encoder], model_type=self.backbone_type)

        if self.init_w0 is None:
            self.init_w0 = 1.0

        if self.init_w1 is None:
            self.init_w1 = [1.0] * len(self.loras)
        elif isinstance(self.init_w1, (int, float)):
            self.init_w1 = [self.init_w1] * len(self.loras)
        elif isinstance(self.init_w1, list):
            if len(self.init_w1) != len(self.loras):
                raise ValueError(f"Length of init_w1 ({len(self.init_w1)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for init_w1: {type(self.init_w1)}")

        self.logger.info(f"Initial w0: {logging.yellow(self.init_w0)}")
        self.logger.info(f"Initial w1: {logging.yellow(self.init_w1)}")

        wrapper_models = lora_utils.wrap_loras_to_model(
            {
                'nnet': self.nnet,
                'text_encoder': self.text_encoder,
            },
            init_w0=self.init_w0,
            init_w1=self.init_w1,
            lora_state_dicts=self.loras,
            model_type=self.backbone_type,
            lora_name_to_module=self.lora_name_to_orig_module,
            lora_name_to_module_name=self.lora_name_to_orig_module_name,
            lora_wrapper_class=self.lora_adapter_class,
            inplace=False,
            verbose=True,
        )

        # cache maps
        self.lora_name_to_module = lora_utils.make_lora_name_to_lora_wrapper_map(wrapper_models.values(), model_type=self.backbone_type, debug_te=False)
        # self.logger.debug(f"lora_name_to_module: {json.dumps({k: v.__class__.__name__ for k, v in self.lora_name_to_module.items()}, indent=2)}")
        # for module in wrapper_models.values():
        #     self.logger.debug(module)
        self.lora_name_to_module_name = lora_utils.make_lora_name_to_module_name_map(wrapper_models.values(), model_type=self.backbone_type)
        self.models_psi = {model_name + '_psi': model for model_name, model in wrapper_models.items()}

        # for lora_sd in self.loras:
        #     del lora_sd
        # device_utils.clean_memory()

        return self.models_psi

    def load_discriminator_model(self):
        from ..models.gan import gan
        discriminator = gan.NLayerDiscriminator().to(self.device)
        discriminator.apply(gan.weights_init)
        return {'discriminator': discriminator}

    def _setup_optims(self):
        super()._setup_optims()

        if self.use_gan:
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

    def setup_phi_params(self):
        self.nnet_phi.requires_grad_(False)
        self.nnet_phi.to(self.device)
        self.nnet_phi.eval()

        for module in self.lora_name_to_module.values():
            module.requires_grad_(False)
            module.to(self.device)

        (
            training_models,
            params_to_optimize,
            self.text_encoder_phi,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder_phi,
            self.learning_rate_te,
            name='text_encoder_phi',
        )
        return [], []

    def setup_psi_params(self):
        self.nnet_psi.requires_grad_(False)
        self.nnet_psi.to(self.device)
        self.nnet_psi.eval()

        (
            _,
            _,
            self.text_encoder_psi,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder_psi,
            self.learning_rate_te,
            name='text_encoder_psi',
        )

        self.lr_w0 = self.lr_w0 if self.lr_w0 is not None else self.learning_rate_nnet
        if self.lr_w1 is None:
            self.lr_w1 = [self.learning_rate] * len(self.loras)
        elif isinstance(self.lr_w1, (int, float)):
            self.lr_w1 = [self.lr_w1] * len(self.loras)
        elif isinstance(self.lr_w1, list):
            if len(self.lr_w1) != len(self.loras):
                raise ValueError(f"Length of lr_w1 ({len(self.lr_w1)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for self: {type(self.lr_w1)}")

        self.logger.info(f"Learning rate for w0: {logging.yellow(self.lr_w0)}")
        self.logger.info(f"Learning rates for w1: {logging.yellow(self.lr_w1)}")

        training_models = [self.nnet_psi, self.text_encoder_psi]
        params_to_optimize = []
        for lora_name, module in self.lora_name_to_module.items():
            module: self.lora_adapter_class
            assert isinstance(module, self.lora_adapter_class), f"Expect module to be {self.lora_adapter_class.__name__}, but got {module.__class__.__name__}"
            module.w0.to(self.device, dtype=self.weight_dtype)
            module.w1.to(self.device, dtype=self.weight_dtype)
            module.w0.requires_grad_(True)
            module.w1.requires_grad_(True)
            params_to_optimize.append({'params': module.w0, 'lr': self.lr_w0})
            params_to_optimize.extend([{'params': module.w1[i], 'lr': self.lr_w1[i]} for i in range(len(module.w1))])
            # logging.debug(f"[{lora_name}] params: {sum(p.numel() for p in [module.w0, *module.w1.parameters()])}")

        self.nnet_psi.to(self.device, dtype=self.weight_dtype)
        self.nnet_psi = self._prepare_one_model(self.nnet_psi, train=True, name="nnet_psi", transform_model_if_ddp=True)

        self.text_encoder_psi.to(self.device)
        self.text_encoder_psi = self._prepare_one_model(self.text_encoder_psi, train=self.train_text_encoder, name="text_encoder_psi", transform_model_if_ddp=True)

        return training_models, params_to_optimize

    # def setup_lora_params(self):
    #     for lora_state_dict in self.loras:
    #         for k, v in lora_state_dict.items():
    #             v.requires_grad_(False)
    #             lora_state_dict[k] = v.to(self.device, dtype=self.weight_dtype)
    #     return [], []

    def encode_caption_kohya(self, captions, text_encoder):
        input_ids = torch.stack([sd15_train_utils.get_input_ids(caption, self.tokenizer, max_token_length=self.max_token_length) for caption in captions], dim=0)
        input_ids = input_ids.to(self.device)
        encoder_hidden_states = sd15_train_utils.get_hidden_states(
            input_ids, self.tokenizer, text_encoder, weight_dtype=None if not self.full_fp16 else self.weight_dtype,
            v2=self.v2, clip_skip=self.clip_skip, max_token_length=self.max_token_length,
        )
        return encoder_hidden_states

    def is_disc_step(self):
        return self.use_gan and self.train_state.global_step % 2 == 0

    def is_gen_step(self):
        return self.use_gan and self.train_state.global_step % 2 == 1

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

    def kl_divergence(self, reconstructed_tensor, groundtruth_tensor, path_size=8, overlap=True, weight=1000):
        B, C, H, W = reconstructed_tensor.shape
    #   print(f'H: {H}, W: {W}')
        # self.logger.debug(f'H: {H}, W: {W}')
        # self.logger.debug(f'path_size: {path_size}')
        assert H % path_size == 0 and W % path_size == 0

        def kld_loss(reconstructed, groundtruth, patch=True):
            dtype = reconstructed.dtype
            reconstructed = reconstructed.to(dtype=torch.float32)
            groundtruth = groundtruth.to(dtype=torch.float32)
            if patch:
                reconstructed = rearrange(reconstructed, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)
                groundtruth = rearrange(groundtruth, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)

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

    def get_loss(self, model_pred, target, timesteps, batch, is_on=False, **kwargs):
        model_pred = model_pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        if self.is_disc_step():
            logits_real = self.discriminator(target.detach().to(dtype=torch.float32))
            logits_fake = self.discriminator(model_pred.detach().to(dtype=torch.float32))
            d_loss = vae_train_utils.hinge_d_loss(logits_real, logits_fake) * self.lambda_gan
            loss = d_loss

            # self.logger.debug(f"Discriminator loss: {d_loss.item()}")
            self.accelerator_logs.update({
                f"disc_loss/step": d_loss.item(),
            })

        else:
            loss = 0
            mse_loss = super().get_loss(model_pred, target, timesteps, batch)
            loss += mse_loss
            if self.lambda_lpips:
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
                loss += self.lambda_lpips * lpips_loss_batch

            if self.is_gen_step():
                kld_loss = self.kl_divergence(model_pred, target, path_size=self.kld_patch_size, overlap=True, weight=10) * self.lambda_kld
                logits_real = self.discriminator(target)
                logits_fake = self.discriminator(model_pred.to(dtype=torch.float32))
                g_loss = vae_train_utils.hinge_d_loss(logits_real, logits_fake) * self.lambda_gan
                loss += kld_loss + g_loss

                # self.logger.debug(f"MSE loss: {mse_loss.item()}")
                # self.logger.debug(f"Generator loss: {g_loss.item()}")
                # self.logger.debug(f"KL divergence loss: {kld_loss.item()}")

            self.accelerator_logs.update({
                f"mse_loss/step": mse_loss.item(),
                f"kld_loss/step": kld_loss.item() if self.is_gen_step() else None,
                f"g_loss/step": g_loss.item() if self.is_gen_step() else None,
                f"lpips_loss/step": lpips_loss_batch.item() if self.lambda_lpips else None,
            })

        return loss

    # def get_loss(self, model_pred, target, timesteps, batch, is_on=False, **kwargs):
    #     is_disc = self.use_gan and self.train_state.global_step % 2 == 0
    #     is_gen = self.use_gan and self.train_state.global_step % 2 == 1
    #     if is_disc:
    #         real_preds = self.discriminator(target)
    #         fake_preds = self.discriminator(model_pred.detach())
    #         d_loss, avg_real_logits, avg_fake_logits, disc_acc = vae_train_utils.gan_disc_loss(
    #             real_preds, fake_preds, self.gan_disc_type
    #         )

    #         avg_real_logits = torch.tensor(avg_real_logits, device=self.device)
    #         avg_fake_logits = torch.tensor(avg_fake_logits, device=self.device)
    #         # avg_real_logits = vae_train_utils.avg_scalar_over_nodes(avg_real_logits, self.device)
    #         # avg_fake_logits = vae_train_utils.avg_scalar_over_nodes(avg_fake_logits, self.device)

    #         total_d_loss = d_loss.mean()
    #         d_loss_item = total_d_loss.item()
    #         if self.use_lecam:
    #             # penalize the real logits to fake and fake logits to real.
    #             lecam_anchor_real_logits = (
    #                 self.lecam_beta * lecam_anchor_real_logits
    #                 + (1 - self.lecam_beta) * avg_real_logits
    #             )
    #             lecam_anchor_fake_logits = (
    #                 self.lecam_beta * lecam_anchor_fake_logits
    #                 + (1 - self.lecam_beta) * avg_fake_logits
    #             )
    #             lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(2).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()
    #             lecam_loss_item = lecam_loss.item()
    #             total_d_loss = total_d_loss + lecam_loss * self.lecam_loss_weight

    #         loss = total_d_loss

    #         self.logger.debug(f"Discriminator loss: {d_loss_item}")

    #         self.accelerator_logs.update({
    #             f"disc_loss/step": d_loss_item,
    #             f"disc_acc/step": disc_acc,
    #             f"avg_real_logits/step": avg_real_logits,
    #             f"avg_fake_logits/step": avg_fake_logits,
    #             f"lecam_loss/step": lecam_loss_item if self.use_lecam else None,
    #         })

    #     else:
    #         loss = 0
    #         mse_loss = super().get_loss(model_pred, target, timesteps, batch)
    #         loss += mse_loss
    #         if self.lambda_lpips:
    #             try:
    #                 if target.shape[1] < 3:
    #                     # We'll put zeros in the third channel...
    #                     n_to_add = 3-target.shape[1]
    #                     target_pad = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, n_to_add), mode='constant', value=0)
    #                     model_pred_pad = torch.nn.functional.pad(model_pred, (0, 0, 0, 0, 0, n_to_add), mode='constant', value=0)
    #                     lpips_loss_batch = self.lpips_loss_fn(model_pred_pad, target_pad).mean()
    #                 elif target.shape[1] > 3:
    #                     lpips_loss_batch = self.lpips_loss_fn(model_pred[:, :3, :, :], target[:, :3, :, :]).mean()
    #                 else:
    #                     lpips_loss_batch = self.lpips_loss_fn(model_pred, target).mean()
    #             except Exception as e:
    #                 self.logger.error(f"Error in calculating LPIPS loss: {e}. Model pred shape: {model_pred.shape}, target shape: {target.shape}")
    #                 lpips_loss_batch = torch.tensor(0.0, device=self.device, dtype=self.weight_dtype)
    #             loss += self.lambda_lpips * lpips_loss_batch

    #         if is_gen:
    #             recon_for_gan = vae_train_utils.gradnorm(model_pred, weight=1.0)
    #             fake_preds = self.discriminator(recon_for_gan)
    #             # real_preds_const = real_preds.clone().detach()
    #             # loss where (real > fake + 0.01)
    #             # g_gan_loss = (real_preds_const - fake_preds - 0.1).relu().mean()
    #             if self.gan_disc_type == "bce":
    #                 g_gan_loss = nn.functional.binary_cross_entropy_with_logits(
    #                     fake_preds, torch.ones_like(fake_preds)
    #                 )
    #             elif self.gan_disc_type == "hinge":
    #                 g_gan_loss = -fake_preds.mean()
    #             g_gan_loss = g_gan_loss.item()
    #         else:
    #             g_gan_loss = 0.0

    #         loss += g_gan_loss

    #         self.logger.debug(f"Generator loss: {g_gan_loss}")

    #         self.accelerator_logs.update({
    #             f"gen_loss/step": g_gan_loss,
    #             f"mse_loss/step": mse_loss.item(),
    #             f"lpips_loss/step": lpips_loss_batch.item() if self.lambda_lpips else None,
    #         })

    #     return loss

    def setup_lpips(self):
        if self.lambda_lpips:
            import lpips
            self.lpips_loss_fn = lpips.LPIPS(net="alex").to(self.accelerator.device)

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

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            z_0 = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                z_0 = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        z_0 *= self.vae_scale_factor

        # encode conditions
        texts = batch['captions']
        texts_phi = [', '.join(self.taus_phi) + ', ' + txt for txt in texts]
        texts_tau = [', '.join(self.taus_psi) + ', ' + txt for txt in texts]
        with self.accelerator.autocast():
            y_theta = self.encode_caption_kohya(texts, self.text_encoder)  # theta
            y_psi = self.encode_caption_kohya(texts, self.text_encoder_psi)  # psi without tau

            y_phi = self.encode_caption_kohya(texts_phi, self.text_encoder_phi)  # phi
            y_psi_tau = self.encode_caption_kohya(texts_tau, self.text_encoder_psi)  # psi with tau

        noise = self.get_noise(z_0)
        t = self.get_timesteps(z_0)
        z_t = self.get_noisy_latents(z_0, noise, t).to(self.weight_dtype)

        with self.accelerator.autocast():
            p_theta = self.nnet(z_t, t, y_theta).sample  # theta
            p_psi = self.nnet_psi(z_t, t, y_psi).sample  # psi without tau

            p_phi = self.nnet_phi(z_t, t, y_phi).sample  # phi
            p_psi_tau = self.nnet_psi(z_t, t, y_psi_tau).sample  # psi with tau

        loss_off = self.get_loss(p_theta, p_psi, timesteps=t, batch=batch, is_on=False)
        loss_on = self.get_loss(p_phi, p_psi_tau, timesteps=t, batch=batch, is_on=True)
        loss = self.loss_beta_1 * loss_on + self.loss_beta_2 * loss_off

        # Debug merge weights W
        W = [module.w1[0] for module in self.lora_name_to_module.values()]
        W = W[0].float().detach().cpu().numpy()
        self.logger.debug(f"step={self.train_state.global_step:5d} | ratios={logging.blue(W)}", write=True)

        self.accelerator_logs.update({
            "loss_on/step":  loss_on.item(),
            "loss_off/step": loss_off.item(),
        })
        self.pbar_logs.update({'loss_on': loss_on.item(), 'loss_off': loss_off.item()})

        return loss
