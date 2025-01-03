import torch
from torch import nn
from ml_collections import ConfigDict
from typing import Dict, List, Union, Literal
from safetensors.torch import load_file
from waifuset import logging
from .sd15_trainer import SD15Trainer
from ..utils import lora_utils, sd15_train_utils, vae_train_utils
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
    lecam_loss_weight = 0.1
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9
    lr_discriminator: float = 1e-3

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
        discriminator = vae_train_utils.PatchDiscriminator().cuda()
        return {'discriminator': discriminator}

    def setup_discriminator_params(self):
        self.discriminator.requires_grad_(True)
        self.discriminator.to(self.device)
        return [self.discriminator], [{'params': self.discriminator.parameters(), 'lr': self.lr_discriminator}]

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

    def encode_caption(self, captions, text_encoder):
        input_ids = torch.stack([sd15_train_utils.get_input_ids(caption, self.tokenizer, max_token_length=self.max_token_length) for caption in captions], dim=0)
        input_ids = input_ids.to(self.device)
        encoder_hidden_states = sd15_train_utils.get_hidden_states(
            input_ids, self.tokenizer, text_encoder, weight_dtype=None if not self.full_fp16 else self.weight_dtype,
            v2=self.v2, clip_skip=self.clip_skip, max_token_length=self.max_token_length,
        )
        return encoder_hidden_states

    def get_loss(self, model_pred, target, timesteps, batch):
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

        if self.use_gan:
            real_preds = self.discriminator(target)
            fake_preds = self.discriminator(model_pred.detach())
            d_loss, avg_real_logits, avg_fake_logits, disc_acc = vae_train_utils.gan_disc_loss(
                real_preds, fake_preds, self.gan_disc_type
            )

            avg_real_logits = vae_train_utils.avg_scalar_over_nodes(avg_real_logits, self.device)
            avg_fake_logits = vae_train_utils.avg_scalar_over_nodes(avg_fake_logits, self.device)

            lecam_anchor_real_logits = (
                self.lecam_beta * lecam_anchor_real_logits
                + (1 - self.lecam_beta) * avg_real_logits
            )
            lecam_anchor_fake_logits = (
                self.lecam_beta * lecam_anchor_fake_logits
                + (1 - self.lecam_beta) * avg_fake_logits
            )
            total_d_loss = d_loss.mean()
            d_loss_item = total_d_loss.item()
            if self.use_lecam:
                # penalize the real logits to fake and fake logits to real.
                lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(
                    2
                ).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()
                lecam_loss_item = lecam_loss.item()
                total_d_loss = total_d_loss + lecam_loss * self.lecam_loss_weight

        if self.use_gan:
            recon_for_gan = vae_train_utils.gradnorm(model_pred, weight=1.0)
            fake_preds = self.discriminator(recon_for_gan)
            real_preds_const = real_preds.clone().detach()
            # loss where (real > fake + 0.01)
            # g_gan_loss = (real_preds_const - fake_preds - 0.1).relu().mean()
            if self.gan_disc_type == "bce":
                g_gan_loss = nn.functional.binary_cross_entropy_with_logits(
                    fake_preds, torch.ones_like(fake_preds)
                )
            elif self.gan_disc_type == "hinge":
                g_gan_loss = -fake_preds.mean()

            g_gan_loss = g_gan_loss.item()
        else:
            g_gan_loss = 0.0

        return loss

    def setup_lpips(self):
        if self.lambda_lpips:
            import lpips
            self.lpips_loss_fn = lpips.LPIPS(net="alex").to(self.accelerator.device)

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
            y_theta = self.encode_caption(texts, self.text_encoder)  # theta
            y_psi = self.encode_caption(texts, self.text_encoder_psi)  # psi without tau

            y_phi = self.encode_caption(texts_phi, self.text_encoder_phi)  # phi
            y_psi_tau = self.encode_caption(texts_tau, self.text_encoder_psi)  # psi with tau

        noise = self.get_noise(z_0)
        t = self.get_timesteps(z_0)
        z_t = self.get_noisy_latents(z_0, noise, t).to(self.weight_dtype)

        with self.accelerator.autocast():
            p_theta = self.nnet(z_t, t, y_theta).sample  # theta
            p_psi = self.nnet_psi(z_t, t, y_psi).sample  # psi without tau

            p_phi = self.nnet_phi(z_t, t, y_phi).sample  # phi
            p_psi_tau = self.nnet_psi(z_t, t, y_psi_tau).sample  # psi with tau

        loss_off = self.get_loss(p_theta, p_psi, timesteps=t, batch=batch)
        loss_on = self.get_loss(p_phi, p_psi_tau, timesteps=t, batch=batch)
        loss = self.loss_beta_1 * loss_on + self.loss_beta_2 * loss_off

        # Debug merge weights W
        W = [module.w1[0] for module in self.lora_name_to_module.values()]
        W = W[0].float().detach().cpu().numpy()
        self.logger.debug(f"step={self.train_state.global_step:5d} | ratios={logging.blue(W)}", write=True)

        self.accelerator_logs.update({"lora_loss/step":  loss_on.item(), 'orig_loss/step': loss_off.item()})
        self.pbar_logs.update({'lora_loss': loss_on.item(), 'orig_loss': loss_off.item()})

        return loss
