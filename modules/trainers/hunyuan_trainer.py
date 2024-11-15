import torch
import os
import gc
from argparse import Namespace
from typing import Literal, Union, Tuple
from waifuset import logging

from modules.datasets.base_dataset import BaseDataset
# from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline
from .sd15_trainer import SD15Trainer
from ..train_state.hunyuan_train_state import HunyuanTrainState
from ..models.hunyuan.modules.models import HunYuanDiT
from ..utils import hunyuan_model_utils
from ..datasets.hunyuan_dataset import HunyuanDataset
from ..models.hunyuan.diffusion.pipeline import StableDiffusionPipeline as HunyuanDiTPipeline

logging.warning(f"HunyuanTrainer is still in development, please use with caution!")


class HunyuanTrainer(SD15Trainer):
    nnet_class = HunYuanDiT
    pipeline_class = HunyuanDiTPipeline
    train_state_class = HunyuanTrainState
    dataset_class = HunyuanDataset

    nnet_model_name_or_path: str = None

    model_type: Literal["DiT-g/2", "DiT-XL/2"] = "DiT-g/2"
    beta_start: float = 0.00085
    beta_end: float = 0.03
    learn_sigma: bool = True
    sigma_small: bool = False
    infer_mode: Literal["fa", "torch", "trt"] = "fa"
    text_states_dim: int = 1024
    text_states_dim_t5: int = 2048
    t5xxl_max_token_length: int = 256
    norm: Literal["rms", "laryer"] = "layer"
    qk_norm: bool = True
    prediction_type: Literal['v_prediction'] = 'v_prediction'
    noise_scheduler_type: Literal['scaled_linear'] = 'scaled_linear'
    mse_loss_weight_type: Literal['constant'] = 'constant'
    size_cond: Union[int, Tuple[int, int]] = None
    use_style_cond: bool = False
    use_flash_attn: bool = False
    use_diffusion: bool = False
    dropout_t5: bool = False

    def get_dataset(self, setup=True, **kwargs) -> BaseDataset:
        kwargs.update(
            latents_dtype=self.weight_dtype,
            patch_size=self.nnet.patch_size,
            hidden_size=self.nnet.hidden_size,
            num_heads=self.nnet.num_heads,
        )
        return super().get_dataset(setup, **kwargs)

    def get_vae_scale_factor(self):
        return self.vae.config.scaling_factor

    def load_tokenizer_model(self):
        # tokenizers are loaded from `load_diffusion_model`
        return {}

    def load_diffusion_model(self):
        # if os.path.isfile(self.pretrained_model_name_or_path):
        #     diffusion_models = hunyuan_model_utils.load_models_from_hunyuan_checkpoint(
        #         self.pretrained_model_name_or_path,
        #         device=self.device,
        #         dtype=torch.float16,
        #     )
        # else:
        #     diffusion_models = hunyuan_model_utils.load_models_from_hunyuan_diffusers_state(
        #         self.pretrained_model_name_or_path,
        #         revision=self.revision,
        #         variant=self.variant,
        #         device=self.device,
        #         dtype=torch.float16,
        #         cache_dir=self.hf_cache_dir,
        #         dropout_t5=self.dropout_t5,
        #         max_retries=self.max_retries,
        #     )
        model_args = Namespace(
            learn_sigma=self.learn_sigma,
            text_states_dim=self.text_states_dim,
            text_states_dim_t5=self.text_states_dim_t5,
            text_len=self.max_token_length,
            text_len_t5=self.t5xxl_max_token_length,
            infer_mode=self.infer_mode,
            norm=self.norm,
            qk_norm=self.qk_norm,
            use_flash_attn=self.use_flash_attn,
            use_fp16=self.weight_dtype == torch.float16,
            size_cond=self.size_cond,
            use_style_cond=self.use_style_cond,
        )

        diffusion_models = hunyuan_model_utils.load_models_from_hunyuan_official(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            model_args=model_args,
            input_size=(128, 128),
            model_type=self.model_type,
            max_token_length_t5=self.t5xxl_max_token_length,
            use_deepspeed=self.use_deepspeed,
            deepspeed_config=self.get_deepspeed_config(),
            deepspeed_remote_device=self.deepspeed_remote_device,
            deepspeed_zero_stage=self.zero_stage,
        )

        if self.nnet_model_name_or_path is not None:
            from safetensors.torch import load_file
            self.logger.info(f"load nnet checkpoint {self.nnet_model_name_or_path}")
            nnet_sd = load_file(self.nnet_model_name_or_path)
            diffusion_models['nnet'].load_state_dict(nnet_sd, strict=True)

        return diffusion_models

    def enable_xformers(self):
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("Please install xformers to use the xformers model")
        self.nnet.set_use_memory_efficient_attention_xformers(True, False)
        if torch.__version__ >= "2.0.0":
            self.vae.set_use_memory_efficient_attention_xformers(True)

    def setup_nnet_params(self):
        training_models, params_to_optimize = super().setup_nnet_params()
        if self.full_fp16:
            from ..models.hunyuan.modules.fp16_layers import Float16Module
            self.nnet = Float16Module(self.nnet, None)
        return training_models, params_to_optimize

    def setup_ema(self):
        from ..models.hunyuan.modules.ema import EMA
        ema_model_args = Namespace(
            ema_dtype='fp16' if self.mixed_precision == 'fp16' else 'bf16' if self.mixed_precision == 'bf16' else 'none',
            use_fp16=self.weight_dtype == torch.float16,
            ema_decay=None,
            ema_warmup=False,
            ema_warmup_power=None,
            ema_reset_decay=False,
        )
        ema = EMA(ema_model_args, model=self.nnet, device=self.device, logger=self.logger)
        ema_path = self.ema_path or os.path.join(self.pretrained_model_name_or_path, 'model', 'pytorch_model_ema.pt')
        if not os.path.exists(ema_path):
            raise FileNotFoundError(f"Cannot find ema checkpoint from {ema_path}")
        self.logger.info(f"resume from ema checkpoint {ema_path}")
        ema_sd = torch.load(ema_path, map_location=lambda storage, loc: storage)
        ema_sd = ema_sd['ema'] if 'ema' in ema_sd.keys() else ema_sd['module'] if 'module' in ema_sd.keys() else ema_sd
        ema.load_state_dict(ema_sd, strict=True)
        ema.eval()
        self.ema = ema

    def setup_text_encoder_params(self):
        training_models = []
        params_to_optimize = []

        (
            training_models_,
            params_to_optimize_,
            self.text_encoder1,
            self.train_text_encoder1,
            self.learning_rate_te1
        ) = self._setup_one_text_encoder_params(
            self.text_encoder1,
            self.learning_rate_te1,
            name='text_encoder1',
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)

        if self.dropout_t5:
            self.logger.print(f"dropout t5")
            encoder_hidden_states_5, text_embedding_mask_5 = self._encode_caption_with_t5([''])
            self.empty_encoder_hidden_states_t5 = encoder_hidden_states_5
            self.empty_text_embedding_mask_t5 = text_embedding_mask_5

            # move t5 to cpu
            self.text_encoder2 = self.text_encoder2.to('cpu')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        else:
            (
                training_models_,
                params_to_optimize_,
                self.text_encoder2,
                self.train_text_encoder2,
                self.learning_rate_te2
            ) = self._setup_one_text_encoder_params(
                self.text_encoder2,
                self.learning_rate_te2,
                name='text_encoder2',
            )
            training_models.extend(training_models_)
            params_to_optimize.extend(params_to_optimize_)

        return training_models, params_to_optimize

    def _setup_diffusion(self):
        if self.use_diffusion:
            self.diffusion = self.get_diffusion()
        else:
            return super()._setup_diffusion()

    def get_diffusion(self):
        from ..models.hunyuan.diffusion import create_diffusion
        diffusion = create_diffusion(
            steps=self.max_timestep,
            learn_sigma=self.learn_sigma,
            sigma_small=self.sigma_small,
            noise_schedule=self.noise_scheduler_type,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            predict_type=self.prediction_type,
            noise_offset=self.noise_offset,
        )
        return diffusion

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
    #         text_encoder=[self.text_encoder1, self.text_encoder2],
    #         tokenizer=[self.tokenizer1, self.tokenizer2],
    #         vae=self.vae,
    #     )

    def ema_step(self):
        nnet = self.nnet if not self.use_deepspeed else self.ds_model.get_models()['nnet'].module
        nnet = nnet.module if self.mixed_precision == 'fp16' else nnet
        self.ema.update(nnet, step=self.train_state.global_step)

    def train_step(self, batch):
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(dtype=self.vae_dtype, device=self.vae.device)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        with torch.set_grad_enabled(self.train_text_encoder):
            encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5 = self.encode_caption(
                batch["captions"],
            )
            if self.full_fp16:
                encoder_hidden_states = encoder_hidden_states.to(self.weight_dtype)
                encoder_hidden_states_t5 = encoder_hidden_states_t5.to(self.weight_dtype)

        # B, C, H, W = noisy_latents.shape
        # freqs_cis_img = hunyuan_train_utils.calc_rope(H * 8, W * 8, 2, 88 if self.model_type == "DiT-g/2" else 72)

        # self.logger.debug(f"freqs_cis_img: {freqs_cis_img[0].shape}, {freqs_cis_img[1].shape}")
        # self.logger.debug(f"cos_cis_img: {batch['cos_cis_img'].shape}, sin_cis_img: {batch['sin_cis_img'].shape}")
        # cos_cis_img, sin_cis_img = freqs_cis_img
        # self.logger.print(f"cos_cis_img.shape: {cos_cis_img.shape}, sin_cis_img.shape: {sin_cis_img.shape}")
        # self.logger.print(f"batch['cos_cis_img'].shape: {batch['cos_cis_img'].shape}, batch['sin_cis_img'].shape: {batch['sin_cis_img'].shape}")

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
            "return_dict": self.use_diffusion,
        }

        nnet = self.nnet if not self.use_deepspeed else self.ds_model.get_models()['nnet']
        if self.use_diffusion:
            with self.accelerator.autocast():
                loss_dict = self.diffusion.training_losses(model=nnet, x_start=latents, model_kwargs=model_kwargs)
                loss = loss_dict["loss"].mean()
        else:
            noise = self.get_noise(latents)
            timesteps = self.get_timesteps(latents)
            noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

            with self.accelerator.autocast():
                noise_pred = nnet(
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

    def _encode_caption_with_bert(self, caption):
        pad_num = 0
        text_input_ids_list = []
        attention_mask_list = []
        for cp in caption:
            text_inputs = self.tokenizer1(
                cp,
                padding="max_length",
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids[0]
            attention_mask = text_inputs.attention_mask[0].bool()
            if pad_num > 0:
                attention_mask[1:pad_num + 1] = False

            # text_input_ids = text_input_ids.clone().detach()
            # attention_mask = attention_mask.clone().detach()

            text_input_ids_list.append(text_input_ids)
            attention_mask_list.append(attention_mask)

        text_embedding = torch.stack(text_input_ids_list, dim=0).to(self.device)
        text_embedding_mask = torch.stack(attention_mask_list, dim=0).to(self.device)
        # self.logger.debug(f"text_encoder1.device: {self.text_encoder1.device}, text_embedding.device: {text_embedding.device}, text_embedding_mask.device: {text_embedding_mask.device}")
        encoder_hidden_states = self.text_encoder1(
            text_embedding,
            attention_mask=text_embedding_mask,
        )[0]
        return encoder_hidden_states, text_embedding_mask

    def fill_t5_token_mask(self, fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat((fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1)
        return fill_tensor

    def _encode_caption_with_t5(self, caption, attention_mask=True, layer_index=-1):
        text_input_ids_t5_list = []
        attention_mask_t5_list = []
        for cp in caption:
            text_tokens_and_mask = self.tokenizer2(
                cp,
                max_length=self.t5xxl_max_token_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            text_input_ids_t5 = self.fill_t5_token_mask(text_tokens_and_mask["input_ids"], fill_number=1, setting_length=self.t5xxl_max_token_length).long()
            attention_mask_t5 = self.fill_t5_token_mask(text_tokens_and_mask["attention_mask"], fill_number=0, setting_length=self.t5xxl_max_token_length).bool().to(self.device)

            text_input_ids_t5_list.append(text_input_ids_t5)
            attention_mask_t5_list.append(attention_mask_t5)
        text_embedding_t5 = torch.cat(text_input_ids_t5_list, dim=0).to(self.device)
        text_embedding_mask_t5 = torch.cat(attention_mask_t5_list, dim=0).to(self.device)
        # self.logger.debug(f"text_encoder2.device: {self.text_encoder2.device}, text_embedding_t5.device: {text_embedding_t5.device}, text_embedding_mask_t5.device: {text_embedding_mask_t5.device}")
        # self.logger.debug(f"text_embedding_t5.shape: {text_embedding_t5.shape}, text_embedding_mask_t5.shape: {text_embedding_mask_t5.shape}")
        with torch.no_grad():
            output_t5 = self.text_encoder2(
                input_ids=text_embedding_t5,
                attention_mask=text_embedding_mask_t5 if attention_mask else None,
                output_hidden_states=True
            )
            encoder_hidden_states_t5 = output_t5['hidden_states'][layer_index].detach()
        return encoder_hidden_states_t5, text_embedding_mask_t5

    def encode_caption(self, captions, attention_mask=True, layer_index=-1):
        encoder_hidden_states, text_embedding_mask = self._encode_caption_with_bert(captions)
        if self.dropout_t5:
            bs = len(captions)
            encoder_hidden_states_t5 = self.empty_encoder_hidden_states_t5.clone().detach().expand(bs, -1, -1).to(self.device)
            text_embedding_mask_t5 = self.empty_text_embedding_mask_t5.clone().detach().expand(bs, -1).to(self.device)
        else:
            encoder_hidden_states_t5, text_embedding_mask_t5 = self._encode_caption_with_t5(captions, attention_mask, layer_index)
        # self.logger.debug(f"shapes: {encoder_hidden_states.shape}, {text_embedding_mask.shape}, {encoder_hidden_states_t5.shape}, {text_embedding_mask_t5.shape}")
        return encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5
