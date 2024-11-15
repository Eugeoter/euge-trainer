import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz
from ml_collections import ConfigDict
from typing import Dict, List, Union
from safetensors.torch import load_file
from waifuset import logging
from .sd15_trainer import SD15Trainer
from ..utils import lora_utils, sd15_train_utils, device_utils
# from ..models.sd15.load_adapter import ConstantLoRAAdapter, MultiLoRAAdapter, HadamardMultiLoRAAdapter, HadamardMultiLoRAWrapper
from ..models.sd15.hadamard_multi_lora_adapter import HadamardMultiLoRAWrapper
from ..train_state.sd15_lora_adapter_train_state import SD15LoRAAdapterTrainState


class SD15LoraAdapterTrainer(SD15Trainer):
    model_type: str = 'sd15'

    # (i) path to a single lora model, (ii) list of paths to multiple lora models, or (iii) dictionary mapping model path to lora trigger word
    pretrained_lora_model_name_or_path: Union[str, Dict[str, str], List[str], List[Dict[str, str]]] = None

    train_state_class = SD15LoRAAdapterTrainState

    nnet_with_lora: torch.nn.Module
    text_encoder_with_lora: torch.nn.Module
    nnet_with_merge: torch.nn.Module
    text_encoder_with_merge: torch.nn.Module

    lora_wrapper_class = HadamardMultiLoRAWrapper
    learning_rate_module_ratio: float = None
    learning_rate_lora_ratios: List[float] = None

    lora_loss_scale: float = 1.0
    orig_loss_scale: float = 1.0

    loras: List[Dict[str, torch.Tensor]]
    lora_trigger_words: List[str]
    lora_taus: List[str]

    init_module_ratio: float = None
    init_merge_ratios: List[float] = None
    init_lora_ratios: List[float] = None

    def get_model_loaders(self):
        loaders = super().get_model_loaders()
        tails = [self.load_lora_model, self.load_with_lora_model, self.load_with_merge_model]
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
        lora_trigger_words = []
        lora_taus = []
        for i, lora_dict in enumerate(lora_dicts):
            lora_path = lora_dict['path']
            trigger_word = lora_dict.get('trigger', '')
            tau = lora_dict.get('tau', '')

            self.logger.info(f"Loading the {i}-th LoRA model from {logging.yellow(lora_path)}")
            lora_state_dict = load_file(lora_path)

            loras.append(lora_state_dict)
            lora_trigger_words.append(trigger_word)
            lora_taus.append(tau)

        self.lora_name_to_weight_shape = lora_utils.make_lora_name_to_weight_shape_map(loras, model_type=self.model_type)

        return {'loras': loras, 'lora_trigger_words': lora_trigger_words, 'lora_taus': lora_taus}

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

    def load_with_lora_model(self):
        if not hasattr(self, 'nnet'):
            raise ValueError("nnet is not loaded yet, please load nnet first.")
        if not hasattr(self, 'text_encoder'):
            raise ValueError("text_encoder is not loaded yet, please load text_encoder first.")
        if not hasattr(self, 'loras'):
            raise ValueError("loras is not loaded yet, please load loras first.")

        if self.init_lora_ratios is None:
            self.init_lora_ratios = [1.0] * len(self.loras)
        elif isinstance(self.init_lora_ratios, (int, float)):
            self.init_lora_ratios = [self.init_lora_ratios] * len(self.loras)
        elif isinstance(self.init_lora_ratios, list):
            if len(self.init_lora_ratios) != len(self.loras):
                raise ValueError(f"Length of init_lora_ratios ({len(self.init_lora_ratios)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for init_lora_ratios: {type(self.init_lora_ratios)}")

        self.logger.info(f"Initial LoRA ratios: {logging.yellow(self.init_lora_ratios)}")

        models_with_lora = lora_utils.merge_loras_to_model(
            {
                'nnet': self.nnet,
                'text_encoder': self.text_encoder,
            },
            lora_state_dicts=self.loras,
            lora_ratios=self.init_lora_ratios,
            model_type=self.model_type,
            merge_device=self.device,
            merge_dtype=self.weight_dtype,
            # name_to_module=self.lora_name_to_module,
            inplace=False,
        )
        self.models_with_lora = {model_name + '_with_lora': model for model_name, model in models_with_lora.items()}

        return self.models_with_lora

    def load_with_merge_model(self):
        r"""
        Initialize models with merge by wrapping lora to model. Ratios are calculated by the initialized LoRA adapter.
        """
        if not hasattr(self, 'nnet'):
            raise ValueError("nnet_ is not loaded yet, please load nnet first.")
        if not hasattr(self, 'text_encoder'):
            raise ValueError("text_encoder is not loaded yet, please load text_encoder first.")

        self.lora_name_to_orig_module = lora_utils.make_lora_name_to_module_map([self.nnet, self.text_encoder], model_type=self.model_type, debug_te=False)
        self.lora_name_to_orig_module_name = lora_utils.make_lora_name_to_module_name_map([self.nnet, self.text_encoder], model_type=self.model_type)

        if self.init_module_ratio is None:
            self.init_module_ratio = 1.0

        if self.init_merge_ratios is None:
            self.init_merge_ratios = [1.0] * len(self.loras)
        elif isinstance(self.init_merge_ratios, (int, float)):
            self.init_merge_ratios = [self.init_merge_ratios] * len(self.loras)
        elif isinstance(self.init_merge_ratios, list):
            if len(self.init_merge_ratios) != len(self.loras):
                raise ValueError(f"Length of init_merge_ratios ({len(self.init_merge_ratios)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for init_merge_ratios: {type(self.init_merge_ratios)}")

        self.logger.info(f"Initial module ratio: {logging.yellow(self.init_module_ratio)}")
        self.logger.info(f"Initial merge ratios: {logging.yellow(self.init_merge_ratios)}")

        wrapper_models = lora_utils.wrap_loras_to_model(
            {
                'nnet': self.nnet,
                'text_encoder': self.text_encoder,
            },
            init_module_ratio=self.init_module_ratio,
            init_lora_ratios=self.init_merge_ratios,
            lora_state_dicts=self.loras,
            model_type=self.model_type,
            lora_name_to_module=self.lora_name_to_orig_module,
            lora_name_to_module_name=self.lora_name_to_orig_module_name,
            lora_wrapper_class=self.lora_wrapper_class,
            inplace=False,
            verbose=True,
        )

        # cache maps
        self.lora_name_to_module = lora_utils.make_lora_name_to_lora_wrapper_map(wrapper_models.values(), model_type=self.model_type, debug_te=False)
        # self.logger.debug(f"lora_name_to_module: {json.dumps({k: v.__class__.__name__ for k, v in self.lora_name_to_module.items()}, indent=2)}")
        # for module in wrapper_models.values():
        #     self.logger.debug(module)
        self.lora_name_to_module_name = lora_utils.make_lora_name_to_module_name_map(wrapper_models.values(), model_type=self.model_type)
        self.models_with_merge = {model_name + '_with_merge': model for model_name, model in wrapper_models.items()}

        # for lora_sd in self.loras:
        #     del lora_sd
        # device_utils.clean_memory()

        return self.models_with_merge

    def setup_with_lora_params(self):
        self.nnet_with_lora.requires_grad_(False)
        self.nnet_with_lora.to(self.device)
        self.nnet_with_lora.eval()

        for module in self.lora_name_to_module.values():
            module.requires_grad_(False)
            module.to(self.device)

        (
            training_models,
            params_to_optimize,
            self.text_encoder_with_lora,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder_with_lora,
            self.learning_rate_te,
            name='text_encoder_with_lora',
        )
        return [], []

    def setup_with_merge_params(self):
        self.nnet_with_merge.requires_grad_(False)
        self.nnet_with_merge.to(self.device)
        self.nnet_with_merge.eval()

        (
            _,
            _,
            self.text_encoder_with_merge,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder_with_merge,
            self.learning_rate_te,
            name='text_encoder_with_merge',
        )

        self.learning_rate_module_ratio = self.learning_rate_module_ratio if self.learning_rate_module_ratio is not None else self.learning_rate_nnet
        if self.learning_rate_lora_ratios is None:
            self.learning_rate_lora_ratios = [self.learning_rate] * len(self.loras)
        elif isinstance(self.learning_rate_lora_ratios, (int, float)):
            self.learning_rate_lora_ratios = [self.learning_rate_lora_ratios] * len(self.loras)
        elif isinstance(self.learning_rate_lora_ratios, list):
            if len(self.learning_rate_lora_ratios) != len(self.loras):
                raise ValueError(f"Length of learning_rate_lora_ratios ({len(self.learning_rate_lora_ratios)}) does not match the number of LoRAs ({len(self.loras)})")
        else:
            raise ValueError(f"Invalid type for learning_rate_lora_ratios: {type(self.learning_rate_lora_ratios)}")

        self.logger.info(f"Learning rate for module ratio: {logging.yellow(self.learning_rate_module_ratio)}")
        self.logger.info(f"Learning rates for LoRA ratios: {logging.yellow(self.learning_rate_lora_ratios)}")

        training_models = [self.nnet_with_merge, self.text_encoder_with_merge]
        params_to_optimize = []
        for module in self.lora_name_to_module.values():
            assert isinstance(module, self.lora_wrapper_class), f"Expect module to be {self.lora_wrapper_class.__name__}, but got {module.__class__.__name__}"
            module.module_ratio.to(self.device, dtype=self.weight_dtype)
            module.lora_ratios.to(self.device, dtype=self.weight_dtype)
            module.module_ratio.requires_grad_(True)
            module.lora_ratios.requires_grad_(True)
            params_to_optimize.append({'params': module.module_ratio, 'lr': self.learning_rate_module_ratio})
            params_to_optimize.extend([{'params': module.lora_ratios[i], 'lr': self.learning_rate_lora_ratios[i]} for i in range(len(module.lora_ratios))])
            # logging.debug(f"[{lora_name}] params: {sum(p.numel() for p in [module.module_ratio, *module.lora_ratios.parameters()])}")

        self.nnet_with_merge.to(self.device, dtype=self.weight_dtype)
        self.nnet_with_merge = self._prepare_one_model(self.nnet_with_merge, train=True, name="nnet_with_merge", transform_model_if_ddp=True)

        self.text_encoder_with_merge.to(self.device)
        self.text_encoder_with_merge = self._prepare_one_model(self.text_encoder_with_merge, train=self.train_text_encoder, name="text_encoder_with_merge", transform_model_if_ddp=True)

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

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        # encode conditions
        with self.accelerator.autocast():
            merge_encoder_hidden_states_with_tau = self.encode_caption([', '.join(self.lora_taus) + ', ' + caption for caption in batch['captions']], self.text_encoder_with_merge)  # theta
            lora_encoder_hidden_states = self.encode_caption([', '.join(self.lora_trigger_words) + ', ' + caption for caption in batch['captions']],
                                                             self.text_encoder_with_lora)  # theta_0 + delta_theta
            merge_encoder_hidden_states = self.encode_caption(batch['captions'], self.text_encoder_with_merge)  # theta
            orig_encoder_hidden_states = self.encode_caption(batch['captions'], self.text_encoder)  # theta_0

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        with self.accelerator.autocast():
            merge_pred_with_tau = self.nnet_with_merge(noisy_latents, timesteps, merge_encoder_hidden_states_with_tau).sample  # theta
            lora_pred = self.nnet_with_lora(noisy_latents, timesteps, lora_encoder_hidden_states).sample  # theta_0 + delta_theta

            merge_pred = self.nnet_with_merge(noisy_latents, timesteps, merge_encoder_hidden_states).sample  # theta
            orig_pred = self.nnet(noisy_latents, timesteps, orig_encoder_hidden_states).sample  # theta_0

        lora_loss = self.get_loss(lora_pred, merge_pred_with_tau, timesteps=timesteps, batch=batch)
        orig_loss = self.get_loss(orig_pred, merge_pred, timesteps=timesteps, batch=batch)
        loss = self.lora_loss_scale * lora_loss + self.orig_loss_scale * orig_loss

        lora_ratios = [module.lora_ratios[0] for module in self.lora_name_to_module.values()]
        lora_ratios = lora_ratios[0].float().detach().cpu().numpy()
        self.logger.debug(f"step={self.train_state.global_step:5d} | ratios={logging.blue(lora_ratios)}", write=True)

        self.accelerator_logs.update({"lora_loss/step":  lora_loss.item(), 'orig_loss/step': orig_loss.item()})
        self.pbar_logs.update({'lora_loss': lora_loss.item(), 'orig_loss': orig_loss.item()})

        return loss
