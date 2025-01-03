import os
import torch
import gc
import re
import math
import random
from diffusers import DDPMScheduler
from typing import Literal, List, Dict, Any, Union, Callable
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.datasets.base_dataset import BaseDataset
from waifuset import logging
from .base_trainer import BaseTrainer
from ..utils import advanced_train_utils, sd15_model_utils, sd15_train_utils, class_utils, train_utils
from ..train_state.sd15_train_state import SD15TrainState
# from ..pipelines.lpw_pipeline import StableDiffusionLongPromptWeightingPipeline
# from ..models.sd15.nnet import UNet2DConditionModel
from ..datasets.t2i_dataset import T2IDataset


class SD15Trainer(BaseTrainer):
    r"""
    A powerful and essential trainer for training text-to-image models (e.g. stable-diffusion-v1-5).

    Sub-classes of :class:`SD15T2ITrainer` should override methods to customize the training process.
    """

    pretrained_model_name_or_path: str
    vae_model_name_or_path: str = None
    no_half_vae: bool = False
    tokenizer_cache_dir: str = None
    max_token_length: int = None
    revision: str = None
    variant: str = None
    use_safetensors: bool = True

    train_nnet: bool = True
    nnet_trainable_params: List[Union[str, re.Pattern]] = None
    learning_rate_nnet: float = None
    train_text_encoder: bool = False
    learning_rate_te: float = None
    block_lrs: List[float] = None

    ema_path: str = None
    v2: bool = False
    use_xformers: bool = False
    mem_eff_attn: bool = False
    sdpa: bool = False
    clip_skip: int = None
    noise_offset: float = 0
    noise_offset_random_strength: bool = False
    multires_noise_iterations: int = 0
    multires_noise_discount: float = 0.25
    adaptive_noise_scale: float = None
    prediction_type: Literal['epsilon', 'v_prediction'] = 'epsilon'

    loss_type: Literal['l1', 'l2', 'huber', 'smooth_l1', 'cmse', 'ew'] = 'l2'
    huber_schedule: Literal['constant', 'exponential', 'snr'] = 'snr'
    huber_c: float = 0.1
    masked_loss: bool = False

    zero_terminal_snr: bool = False
    desired_terminal_snr: float = None
    ip_noise_gamma: float = None
    min_snr_gamma: float = None
    debiased_estimation_loss: bool = False
    scale_v_pred_loss_like_noise_pred: bool = False
    min_timestep: int = 0
    max_timestep: int = 1000
    max_token_length: int = 225

    use_edm2: bool = False
    edm2_lr: float = 1e-2
    edm2_optimizer_type: str = None
    edm2_optimizer_kwargs: Dict[str, Any] = {}
    edm2_lr_scheduler_type: str = 'auto'
    edm2_lr_scheduler_kwargs: Dict[str, Any] = {}
    edm2_lr_warmup_steps: int = 100
    edm2_lr_constant_steps: int = 300
    edm2_lr_scheduler_num_cycles: int = 1
    edm2_lr_scheduler_power: float = 1.0

    timestep_sampler: Callable[[int, int, int], int] = None
    timestep_sampler_type: Literal['uniform', 'logit_normal', 'auto'] = 'uniform'
    timestep_sampler_kwargs: Dict[str, Any] = class_utils.cfg()

    do_classifier_free_guidance: bool = False
    caption_weighting: bool = False
    max_embeddings_multiples: int = 3
    condition_dropout_rate: float = 0.01

    dataset_class = T2IDataset
    train_dataset: dataset_class
    nnet_class = UNet2DConditionModel
    nnet: nnet_class
    pipeline_class = StableDiffusionPipeline
    pipeline: pipeline_class
    train_state_class = SD15TrainState
    train_state: train_state_class
    text_encoder: torch.nn.Module
    tokenizer: Any
    vae: AutoencoderKL

    cache_latents: bool = False
    cache_only: bool = False
    persistent_data_loader_workers: bool = False
    max_dataloader_n_workers: int = 4
    max_dataset_n_workers: int = 1
    ignore_warnings: bool = True
    loss_recorder_kwargs = class_utils.cfg(
        gamma=0.9,
        stride=1000,
    )
    gc_every_n_steps: int = 1000
    gc_every_n_epochs: int = 1

    weight_dtype: torch.dtype
    vae_dtype: torch.dtype
    save_dtype: torch.dtype

    def get_setups(self):
        if self.cache_latents and self.cache_only:
            setups = [
                self._setup_dtype,
                self._setup_model,
                self._setup_dataset
            ]
        else:
            setups = [
                self._setup_dtype,
                self._setup_model,
                self._setup_diffusion,
                self._setup_dataset,
                self._setup_training,
                self._setup_params,
                self._setup_optims,
                self._setup_loss_recorder,
                self._setup_train_state,
            ]
        return setups

    def check_config(self):
        # TODO: more checks
        if self.full_fp16 and self.mixed_precision != "fp16":
            raise ValueError("full_fp16 requires mixed precision='fp16'")
        if self.full_bf16 and self.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed precision='bf16'")

        if self.prediction_type not in ['epsilon', 'v_prediction']:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        if self.zero_terminal_snr and self.prediction_type != 'v_prediction':
            self.logger.warning("zero_terminal_snr requires prediction_type='v_prediction'")
        if self.prediction_type == 'v_prediction' and self.noise_offset:
            self.logger.warning("noise_offset should not be used for v_prediction")

        if self.use_edm2:
            if self.prediction_type != 'v_prediction':
                self.logger.warning(f"Adaptive loss weighting is designed only for v_prediction, but got prediction_type={self.prediction_type}")
            if self.min_snr_gamma:
                self.logger.warning("Adaptive loss weighting should not be used with min_snr_gamma")
            if self.debiased_estimation_loss:
                self.logger.warning("Adaptive loss weighting should not be used with debiased_estimation_loss")
            if self.scale_v_pred_loss_like_noise_pred:
                self.logger.warning("Adaptive loss weighting should not be used with scale_v_pred_loss_like_noise_pred")
            if self.loss_weight_getter is not None:
                self.logger.warning("Adaptive loss weighting should not be used with loss_weight_getter")
            if self.timestep_sampler_type == "auto":
                self.logger.warning("Adaptive loss weighting should not be used with timestep_sampler_type='auto'")
            if self.use_deepspeed:
                if self.edm2_optimizer_type is not None:
                    raise ValueError("edm2_optimizer_type should be None when using deepspeed")
                if self.edm2_lr_scheduler_type is not None:
                    raise ValueError("edm2_lr_scheduler_type should be None when using deepspeed")

        if sum([self.use_xformers, self.sdpa, self.mem_eff_attn]) > 1:
            raise ValueError("Only one of use_xformers, sdpa, mem_eff_attn can be True")

    def get_dtypes(self):
        dtypes = super().get_dtypes()
        vae_dtype = torch.float32 if self.no_half_vae else dtypes['weight_dtype']
        return {**dtypes, 'vae_dtype': vae_dtype}

    def _setup_model(self):
        super()._setup_model()

        if self.use_xformers:
            self.enable_xformers()
            self.logger.info("enable xformers")
        elif self.sdpa:
            self.enable_sdpa()
            self.logger.info("enable sdpa")
        elif self.mem_eff_attn:
            self.enable_mem_eff_attn()
            self.logger.info("enable memory efficient attention")

    def get_model_loaders(self):
        if self.cache_only:
            return [self.load_vae_model]
        else:
            return super().get_model_loaders()

    def load_vae_model(self):
        if self.vae_model_name_or_path is None:
            return {}
        vae = sd15_model_utils.load_vae(self.vae_model_name_or_path, dtype=self.vae_dtype)
        self.logger.print(f"additional vae model loaded from {self.vae_model_name_or_path}")
        return {'vae': vae}

    def load_tokenizer_model(self):
        tokenizer = sd15_model_utils.load_sd15_tokenizer(
            sd15_model_utils.TOKENIZER_PATH if not self.v2 else sd15_model_utils.V2_STABLE_DIFFUSION_PATH,
            subfolder=None if not self.v2 else 'tokenizer',
            cache_dir=self.tokenizer_cache_dir or self.hf_cache_dir,
            max_token_length=self.max_token_length
        )
        return {'tokenizer': tokenizer}

    def load_diffusion_model(self):
        if os.path.isfile(self.pretrained_model_name_or_path):
            diffusion_models = sd15_model_utils.load_models_from_stable_diffusion_checkpoint(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                v2=self.v2,
                nnet_class=self.nnet_class,
                strict=False,
            )
        else:
            diffusion_models = sd15_model_utils.load_models_from_stable_diffusion_diffusers_state(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                cache_dir=self.hf_cache_dir,
                nnet_class=self.nnet_class,
                max_retries=self.max_retries,
            )
        assert isinstance(diffusion_models['nnet'], self.nnet_class), f"Expected nnet to be {self.nnet_class}, but got {diffusion_models['nnet'].__class__}"
        return diffusion_models

    def enable_xformers(self):
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("Please install xformers to use the xformers model")
        if not self.cache_only:
            # nnet
            if hasattr(self.nnet, 'set_use_memory_efficient_attention'):
                self.nnet.set_use_memory_efficient_attention(True, False)
            elif hasattr(self.nnet, 'set_use_memory_efficient_attention_xformers'):
                self.nnet.set_use_memory_efficient_attention_xformers(True)
            else:
                self.logger.warning(f"xformers seems not supported for this nnet model: {self.nnet.__class__.__name__}")
        # vae
        if torch.__version__ >= "2.0.0":
            self.vae.set_use_memory_efficient_attention_xformers(True)
        else:
            self.logger.warning(f"XFormers not supported for vae in this PyTorch version. 2.0.0+ required, but got: {torch.__version__}")

    def enable_sdpa(self):
        if not self.cache_only:
            if hasattr(self.nnet, 'set_use_sdpa'):
                self.nnet.set_use_sdpa(True)
            else:
                self.logger.warning(f"SDPA seems not supported for this nnet model: {self.nnet.__class__.__name__}")

    def enable_mem_eff_attn(self):
        if not self.cache_only:
            if hasattr(self.nnet, 'set_use_memory_efficient_attention'):
                self.nnet.set_use_memory_efficient_attention(True, True)
            else:
                self.logger.warning(f"Memory efficient attention seems not supported for this nnet model: {self.nnet.__class__.__name__}")

    def get_vae_scale_factor(self):
        return sd15_train_utils.VAE_SCALE_FACTOR

    def _setup_training(self):
        super()._setup_training()

        if self.timestep_sampler_type == 'auto':
            self.loss_for_timesteps = [0]
            self.current_timesteps = [0]
            self.loss_map = {}
            self.run_number = 0

    def get_edm2_mlp(self):
        edm2_mlp = advanced_train_utils.AdaptiveLossWeightMLP(
            self.noise_scheduler,
            logvar_channels=128,
            lambda_weights=None,
        )
        return edm2_mlp

    def _setup_dataset(self):
        super()._setup_dataset()
        if self.cache_latents:
            self.cache_image_latents()

    def get_dataset(self, setup=True, **kwargs) -> BaseDataset:
        kwargs.update(
            latents_dtype=self.weight_dtype,
        )
        return super().get_dataset(setup, **kwargs)

    def cache_image_latents(self):
        with torch.no_grad():
            self.train_dataset.cache_batch_latents(self.vae)
        self.vae.to('cpu')  # vae is not needed anymore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.accelerator.wait_for_everyone()

    def setup_vae_params(self):
        self.vae.to(device=self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.get_vae_scale_factor()
        self.logger.print(f"VAE scale factor: {self.vae_scale_factor}")
        return [], []

    def setup_nnet_params(self):
        training_models = []
        params_to_optimize = []

        learning_rate_nnet = self.learning_rate_nnet or self.learning_rate
        # self.logger.debug(f"learning_rate_nnet: {learning_rate_nnet}")
        train_nnet = self.train_nnet and (learning_rate_nnet > 0 or self.block_lrs is not None)
        if train_nnet:
            if self.gradient_checkpointing:
                self.nnet.enable_gradient_checkpointing()
            training_models.append(self.nnet)
            if self.nnet_trainable_params:  # only train specific parameters
                self.logger.info(f"filtering trainable parameters of nnet by patterns: {self.nnet_trainable_params}")
                if isinstance(self.nnet_trainable_params, (str, re.Pattern)):
                    self.nnet_trainable_params = [self.nnet_trainable_params]
                self.nnet_trainable_params = [re.compile(p) for p in self.nnet_trainable_params]
                nnet_params = []
                self.nnet.requires_grad_(True)
                for name, params in self.nnet.named_parameters():
                    if any(pattern.match(name) for pattern in self.nnet_trainable_params):
                        params.requires_grad = True
                        nnet_params.append(params)
                    else:
                        params.requires_grad = False
                params_to_optimize.append({"params": nnet_params, "lr": learning_rate_nnet})
            elif self.block_lrs is not None:
                self.logger.info(f"applying block learning rates to nnet: {self.block_lrs}")
                self.nnet.requires_grad_(True)
                params_to_optimize = self.setup_block_lrs()
            else:
                self.logger.info(f"training all parameters of nnet")
                self.nnet.requires_grad_(True)
                params_to_optimize.append({"params": list(self.nnet.parameters()), "lr": learning_rate_nnet})
        else:
            self.nnet.requires_grad_(False)

        self.nnet.to(device=self.device, dtype=self.weight_dtype)
        self.nnet = self._prepare_one_model(self.nnet, train=train_nnet, name='nnet', transform_model_if_ddp=True)
        self.train_nnet = train_nnet
        self.learning_rate_nnet = learning_rate_nnet

        return training_models, params_to_optimize

    def setup_block_lrs(self):
        raise NotImplementedError(f"setup_block_lrs is not implemented for {self.__class__.__name__}")

    def get_nnet_trainable_params(self):
        if self.nnet_trainable_params:
            if isinstance(self.nnet_trainable_params, str):
                self.nnet_trainable_params = [self.nnet_trainable_params]
            self.nnet_trainable_params = [re.compile(p) for p in self.nnet_trainable_params]

    def _setup_one_text_encoder_params(self, text_encoder, learning_rate_te, name=None):
        training_models = []
        params_to_optimize = []

        learning_rate_te = learning_rate_te if learning_rate_te is not None else self.learning_rate
        train_text_encoder = self.train_text_encoder and learning_rate_te > 0
        if train_text_encoder:
            if self.gradient_checkpointing:
                text_encoder.gradient_checkpointing_enable()
            text_encoder.requires_grad_(True)
            text_encoder.train()
            training_models.append(text_encoder)
            params_to_optimize.append({"params": list(text_encoder.parameters()), "lr": learning_rate_te})
        else:
            learning_rate_te = 0
            text_encoder.to(self.weight_dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()

        text_encoder = self._prepare_one_model(text_encoder, train=train_text_encoder, name=name, transform_model_if_ddp=True)
        return training_models, params_to_optimize, text_encoder, train_text_encoder, learning_rate_te

    def setup_text_encoder_params(self):
        (
            training_models,
            params_to_optimize,
            self.text_encoder,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder,
            self.learning_rate_te,
            name='text_encoder',
        )
        return training_models, params_to_optimize

    def setup_edm2_mlp_params(self):
        if not self.use_edm2:
            return [], []
        self.edm2_mlp.to(self.device, dtype=torch.float32)
        self.edm2_mlp.requires_grad_(True)
        self.edm2_mlp.train()
        self.edm2_mlp = self._prepare_one_model(self.edm2_mlp, train=True, name='edm2_mlp', dtype=torch.float32, transform_model_if_ddp=True)
        edm2_lr = self.edm2_lr or self.learning_rate
        return [self.edm2_mlp], [{"params": self.edm2_mlp.parameters(), "lr": edm2_lr}] if self.edm2_optimizer_type is None else []

    def _setup_optims(self):
        super()._setup_optims()

        if self.use_edm2:
            if self.edm2_optimizer_type is None:
                self.logger.info(f"Set optimizer of edm2 to {logging.yellow(self.optimizer_type)}")
                self.edm2_optimizer = self.optimizer
            else:
                self.logger.info(f"Use optimizer for adaptive loss weighting: {logging.yellow(self.edm2_optimizer_type)}")
                self.edm2_optimizer = train_utils.get_optimizer(
                    optimizer_type=self.edm2_optimizer_type,
                    trainable_params=self.edm2_mlp.parameters(),
                    lr=self.edm2_lr,
                    lr_scheduler_type=self.edm2_lr_scheduler_type,
                    **self.edm2_optimizer_kwargs,
                )
                self.edm2_optimizer = self.accelerator.prepare(self.edm2_optimizer)

            if self.edm2_lr_scheduler_type is not None:  # use lr_scheduler directly
                self.logger.info(f"Set lr scheduler for adaptive loss weighting to {logging.yellow(self.edm2_lr_scheduler_type)}")
                self.edm2_lr_scheduler = self.lr_scheduler
            elif self.edm2_lr_scheduler_type == 'auto':  # use lr_lambda
                self.logger.info(f"Use auto lr scheduler for adaptive loss weighting")

                def lr_lambda(current_step: int):
                    warmup_steps = self.edm2_lr_warmup_steps
                    constant_steps = self.edm2_lr_constant_steps
                    if current_step <= warmup_steps:
                        return current_step / max(1, warmup_steps)
                    else:
                        return 1 / math.sqrt(max(current_step / (warmup_steps + constant_steps), 1))

                self.edm2_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=self.edm2_optimizer,
                    lr_lambda=lr_lambda
                )
                self.edm2_lr_scheduler = self.accelerator.prepare(self.edm2_lr_scheduler)
            else:  # use edm2_lr_scheduler_type
                self.logger.info(f"Use constant lr scheduler for adaptive loss weighting: {logging.yellow(self.edm2_lr_scheduler_type)}")
                self.edm2_lr_scheduler = train_utils.get_scheduler_fix(
                    lr_scheduler_type=self.edm2_lr_scheduler_type,
                    optimizer=self.edm2_optimizer,
                    num_train_steps=self.num_train_steps,
                    num_warmup_steps=self.edm2_lr_warmup_steps,
                    num_cycles=self.edm2_lr_scheduler_num_cycles,
                    power=self.edm2_lr_scheduler_power,
                    ** self.edm2_lr_scheduler_kwargs,
                )
                self.edm2_lr_scheduler = self.accelerator.prepare(self.edm2_lr_scheduler)

    def _setup_diffusion(self):
        r"""
        Setup components related to diffusion models.
        """
        self.noise_scheduler = self.get_noise_scheduler()

        if self.use_edm2:
            self.edm2_mlp = self.get_edm2_mlp()

    def get_noise_scheduler(self):
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False,
            prediction_type=self.prediction_type,  # rescale_betas_zero_snr=self.zero_terminal_snr,
        )
        if self.prediction_type is not None:  # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=self.prediction_type)
        if self.zero_terminal_snr or self.desired_terminal_snr is not None:
            advanced_train_utils.apply_zero_terminal_snr(noise_scheduler, self.desired_terminal_snr)
        sd15_train_utils.prepare_scheduler_for_custom_training(noise_scheduler, self.device)  # prepare all_snr

        # Check ztsnr
        if self.zero_terminal_snr:
            assert noise_scheduler.all_snr[-1] != 0, f"Expected all_snr[-1] to be 0 when zero_terminal_snr is True, but got {noise_scheduler.all_snr[-1]}"

        return noise_scheduler

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
    #         text_encoder=self.text_encoder,
    #         tokenizer=self.tokenizer,
    #         vae=self.vae,
    #     )

    def get_start_training_message(self):
        messages = super().get_start_training_message()
        timestep_sampler_kwargs_str = ', '.join(f"{k}={v}" for k, v in self.timestep_sampler_kwargs.items())
        messages.append(f"  timestep sampler: {self.timestep_sampler_type}({timestep_sampler_kwargs_str})")
        if self.use_xformers:
            messages.append(f"  use xformers")
        return messages

    def get_loss_weight(self, batch, loss: torch.Tensor):
        with torch.no_grad():  # do not require grad for loss_weight_getter
            # move to cpu to avoid OOM
            loss_weight = self.loss_weight_getter(
                batch=batch,
                dataset_hook=self.train_dataset.dataset_hook,
                loss=loss.detach().cpu(),
                epoch=self.train_state.epoch,
                step=self.train_state.global_step,
            ).to(loss.device)
        if loss_weight.shape != loss.shape:
            raise ValueError(f"loss_weight shape {loss_weight.shape} does not match loss shape {loss.shape}")
        return loss_weight

    def get_loss(self, model_pred, target, timesteps, batch):
        cond_loss_kwargs = {}
        if self.loss_type == 'ew':
            cond_loss_kwargs['alphas_cumprod'] = self.noise_scheduler.alphas_cumprod.to(self.device)
            cond_loss_kwargs['timesteps'] = timesteps
            cond_loss_kwargs['sched_train_steps'] = self.num_train_steps
            cond_loss_kwargs['c_step'] = self.train_state.global_step
        elif self.loss_type == 'cmse':
            cond_loss_kwargs['timesteps'] = timesteps
            cond_loss_kwargs['loss_map'] = self.loss_map
        elif self.loss_type == 'huber':
            cond_loss_kwargs['huber_c'] = self.get_huber_c(timesteps)
        elif self.loss_type == 'smooth_l1':
            cond_loss_kwargs['huber_c'] = self.get_huber_c(timesteps)

        if (
            self.min_snr_gamma or
            self.debiased_estimation_loss or
            self.scale_v_pred_loss_like_noise_pred or
            self.loss_weight_getter is not None or
            self.masked_loss or
            self.timestep_sampler_type == "auto" or
            self.use_edm2
        ):
            loss = train_utils.conditional_loss(
                model_pred.float(),
                target.float(),
                reduction="none",
                loss_type=self.loss_type,
                **cond_loss_kwargs
            )

            if self.masked_loss:
                loss = advanced_train_utils.apply_masked_loss(loss, batch)

            if self.loss_weight_getter is not None:
                loss_weight = self.get_loss_weight(batch, loss)
                loss = loss * loss_weight

            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = loss.mean([1, 2, 3])

            if self.min_snr_gamma:
                loss = advanced_train_utils.apply_snr_weight(loss, timesteps, self.noise_scheduler, self.min_snr_gamma, self.prediction_type)
            if self.scale_v_pred_loss_like_noise_pred:
                loss = advanced_train_utils.scale_v_prediction_loss_like_noise_prediction(loss, timesteps, self.noise_scheduler)
            if self.debiased_estimation_loss:
                loss = advanced_train_utils.apply_debiased_estimation(loss, timesteps, self.noise_scheduler, v_prediction=self.prediction_type == 'v_prediction')

            if self.timestep_sampler_type == "auto":
                self.loss_for_timesteps = loss
                loss = advanced_train_utils.apply_loss_adjustment(loss, timesteps, self.loss_map, self.train_state.global_step, self.num_train_steps)

            if self.use_edm2:
                loss, loss_scaled = self.edm2_mlp(loss, timesteps)
                loss_scaled = loss_scaled.mean()
                self.accelerator_logs.update({"edm2_loss/step":  loss_scaled.detach().item()})
                self.pbar_logs.update({'edm2_loss': loss_scaled.detach().item()})
            loss = loss.mean()
        else:
            loss = train_utils.conditional_loss(
                model_pred.float(),
                target.float(),
                reduction="mean",
                loss_type=self.loss_type,
                **cond_loss_kwargs
            )

        if torch.isnan(loss):
            self.logger.warning(f"Loss is nan at step {self.train_state.global_step}. Replacing with 0.")
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        elif torch.isinf(loss):
            self.logger.warning(f"Loss is inf at step {self.train_state.global_step}. Replacing with 0.")
            loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
        return loss

    def get_noise(self, latents):
        noise = torch.randn_like(latents, device=latents.device)
        if self.noise_offset:
            # code from https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans
            if self.noise_offset_random_strength:
                noise_offset = torch.rand(1, device=latents.device) * self.noise_offset
            else:
                # noise_start = 0.0
                # noise_offset = noise_start + ((self.noise_offset - noise_start) * (self.train_state.global_step / self.num_train_steps))

                # min_offset = 0.5
                # max_offset = 1.5
                # random_offset = random.uniform(min_offset, max_offset)
                # noise_offset = noise_offset * random_offset
                pass
            noise = advanced_train_utils.apply_noise_offset(latents, noise, self.noise_offset, self.adaptive_noise_scale)
        if self.multires_noise_iterations:
            # code from https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans
            min_iter = self.multires_noise_iterations - 3
            if min_iter < 1:
                min_iter = 1
            max_iter = self.multires_noise_iterations + 3

            rand_iter = random.randint(min_iter, max_iter)

            min_discount = self.multires_noise_discount - 0.15
            if min_discount < 0.01:
                min_discount = 0.01
            max_discount = self.multires_noise_discount + 0.15
            if max_discount > 0.99:
                max_discount = 0.99

            rand_discount = random.uniform(min_discount, max_discount)
            noise = advanced_train_utils.pyramid_noise_like(
                noise, latents.device, rand_iter, rand_discount
            )
        return noise

    def get_timesteps(self, latents):
        b_size = latents.shape[0]
        min_timestep = 0 if self.min_timestep is None else self.min_timestep
        max_timestep = self.noise_scheduler.config.num_train_timesteps if self.max_timestep is None else self.max_timestep

        if self.timestep_sampler is not None:
            timesteps = self.timestep_sampler(b_size, min_timestep, max_timestep)
        elif self.timestep_sampler_type == "auto":
            self.loss_map = advanced_train_utils.update_loss_map_ema(self.loss_map, self.loss_for_timesteps, self.current_timesteps)
            timesteps, self.current_timesteps = advanced_train_utils.timestep_attention(self.run_number, self.loss_map, max_timestep, b_size, device=latents.device)
            self.current_timesteps = timesteps
        elif self.timestep_sampler_type == "uniform":
            timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)  # timestep is in [min_timestep, max_timestep)
            timesteps = timesteps.long()
        elif self.timestep_sampler_type == "logit_normal":  # Rectified Flow from SD3 paper (partial implementation)
            timestep_sampler_kwargs = self.timestep_sampler_kwargs
            m = timestep_sampler_kwargs.get('loc', 0) or timestep_sampler_kwargs.get('mean', 0) or timestep_sampler_kwargs.get('m', 0) or timestep_sampler_kwargs.get('mu', 0)
            s = timestep_sampler_kwargs.get('scale', 1) or timestep_sampler_kwargs.get('std', 1) or timestep_sampler_kwargs.get('s', 1) or timestep_sampler_kwargs.get('sigma', 1)
            timesteps = advanced_train_utils.logit_normal(mu=m, sigma=s, shape=(b_size,), device=latents.device)  # sample from logistic normal distribution
            timesteps = timesteps * (max_timestep - min_timestep) + min_timestep  # scale to [min_timestep, max_timestep)
            timesteps = timesteps.long()
        else:
            raise ValueError(f"Unknown timestep sampler type {self.timestep_sampler_type}")

        return timesteps

    def get_huber_c(self, timesteps):
        if self.loss_type == "huber" or self.loss_type == "smooth_l1":
            # timesteps = torch.randint(min_timestep, max_timestep, (1,), device="cpu")
            timesteps = timesteps.tolist()

            for timestep in timesteps:
                # Example processing: calculate a huber_c based on the timestep
                if self.huber_schedule == "exponential":
                    alpha = -math.log(self.huber_c) / self.noise_scheduler.config.num_train_timesteps
                    huber_c = math.exp(-alpha * timestep)
                elif self.huber_schedule == "snr":
                    alphas_cumprod = self.noise_scheduler.alphas_cumprod[timestep]
                    sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
                    huber_c = (1 - self.huber_c) / (1 + sigmas) ** 2 + self.huber_c
                elif self.huber_schedule == "constant":
                    huber_c = self.huber_c
                else:
                    raise NotImplementedError(f"Unknown Huber loss schedule {self.huber_schedule}!")
            # timesteps = torch.tensor(timesteps, device=self.device)

        elif self.loss_type != "huber" or self.loss_type != "smooth_l1":
            # timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=device)
            huber_c = 1  # may be anything, as it's not used
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        return huber_c

    def dropout_condition(self, conditions):
        bs = conditions['crossattn'].shape[0]
        device = conditions['crossattn'].device
        dtype = conditions['crossattn'].dtype

        p = 1.0 - self.condition_dropout_rate
        batch_mask = torch.bernoulli(p * torch.ones(bs, device=device, dtype=dtype))

        for cond_batch in conditions.values():
            cond_batch.mul_(expand_dims_like(batch_mask, cond_batch))

        return conditions

    def get_noisy_latents(self, latents, noise, timesteps):
        if self.ip_noise_gamma:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise + self.ip_noise_gamma * torch.randn_like(latents), timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents

    def encode_caption(self, captions):
        input_ids = torch.stack([sd15_train_utils.get_input_ids(caption, self.tokenizer, max_token_length=self.max_token_length) for caption in captions], dim=0)
        with torch.set_grad_enabled(self.train_text_encoder):
            input_ids = input_ids.to(self.device)
            encoder_hidden_states = sd15_train_utils.get_hidden_states(
                input_ids, self.tokenizer, self.text_encoder, weight_dtype=None if not self.full_fp16 else self.weight_dtype,
                v2=self.v2, clip_skip=self.clip_skip, max_token_length=self.max_token_length,
            )
        return encoder_hidden_states

    def optimizer_step(self, loss):
        super().optimizer_step(loss)
        if self.use_edm2 and self.edm2_optimizer is not self.optimizer:
            self.edm2_optimizer.step()

    def lr_scheduler_step(self):
        super().lr_scheduler_step()
        if self.use_edm2 and self.edm2_lr_scheduler is not self.lr_scheduler:
            self.edm2_lr_scheduler.step()

    def zero_grad(self):
        super().zero_grad()
        if self.use_edm2 and self.edm2_optimizer is not self.optimizer:
            self.edm2_optimizer.zero_grad()

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        encoder_hidden_states = self.encode_caption(batch['captions'])
        if self.condition_dropout_rate:
            encoder_hidden_states = self.dropout_condition(encoder_hidden_states)

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        with self.accelerator.autocast():
            model_pred = self.nnet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps, batch)
        return loss


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x
