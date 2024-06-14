import os
import time
import math
import torch
import gc
from accelerate import Accelerator
from diffusers import DDPMScheduler
from typing import Literal
from ..utils import log_utils, advanced_train_utils, train_utils, model_utils, class_utils
from ..train_state.train_state import TrainState
from ..pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from ..models.original_unet import UNet2DConditionModel
from ..datasets.t2i_dataset import T2ITrainDataset
from ..pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline


class T2ITrainer(class_utils.FromConfigMixin):
    vae_model_name_or_path: str = None
    vae_batch_size: int = 16
    no_half_vae: bool = False
    tokenizer_cache_dir: str = None
    max_token_length: int = None

    block_lr: list = None
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    lr_warmup_steps: int = 0
    lr_scheduler_power: float = 1.0
    lr_scheduler_num_cycles: int = 1
    lr_scheduler_kwargs: dict = class_utils.cfg()
    mixed_precision: Literal['fp16', 'bf16', 'float'] = 'fp16'
    full_fp16: bool = False
    full_bf16: bool = False

    xformers: bool = False
    mem_eff_attn: bool = False
    sdpa: bool = False
    clip_skip: int = None
    noise_offset: float = 0
    multires_noise_iterations: int = 0
    multires_noise_discount: float = 0.25
    adaptive_noise_scale: float = None
    max_grad_norm: float = 0.5
    prediction_type: Literal['epsilon', 'velocity'] = 'epsilon'
    zero_terminal_snr: bool = False
    ip_noise_gamma: float = None
    min_snr_gamma: float = None
    debiased_estimation_loss: bool = False
    min_timestep: int = 0
    max_timestep: int = 1000
    max_token_length: int = 225
    timestep_sampler_type: str = 'uniform'
    timestep_sampler_kwargs: dict = class_utils.cfg()
    cpu: bool = False

    hf_cache_dir: str = None
    hf_token: str = None
    max_retries: int = None

    dataset_class = T2ITrainDataset
    nnet_class = UNet2DConditionModel
    pipeline_class = StableDiffusionLongPromptWeightingPipeline
    train_state_class = TrainState

    persistent_data_loader_workers: bool = False
    max_dataloader_n_workers: int = 4
    max_dataset_n_workers: int = 1
    ignore_warnings: bool = True
    loss_recorder_kwargs = class_utils.cfg(
        gamma=0.9,
        stride=1000,
    )

    def setup(self):
        setups = [
            self._setup_accelerator,
            self._setup_dtype,
            self._setup_model,
            self._setup_dataset,
            self._setup_training,
            self._setup_params,
            self._setup_optims,
            self._setup_noise_scheduler,
            self._setup_loss_recorder,
            self._setup_train_state,
        ]
        for setup in setups:
            setup()

    def _setup_accelerator(self):
        self.accelerator = self.get_accelerator()
        self.device = self.accelerator.device
        self.logger = self.get_logger()
        if self.ignore_warnings:
            train_utils.ignore_warnings()

    def get_accelerator(self):
        log_dir = os.path.join(self.output_dir, self.output_subdir.logs)
        log_dir = log_dir + "/" + time.strftime("%Y%m%d%H%M%S", time.localtime())
        os.makedirs(log_dir, exist_ok=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with='tensorboard',
            project_dir=log_dir,
            cpu=self.cpu,
        )
        return accelerator

    def get_logger(self):
        logger = log_utils.get_logger("train", disable=not self.accelerator.is_main_process)
        for lg in log_utils.get_all_loggers().values():
            lg.disable = not self.accelerator.is_main_process
        return logger

    def _setup_dtype(self):
        dtypes = self.get_dtypes()
        for key, dtype in dtypes.items():
            self.__dict__[key] = dtype

    def get_dtypes(self):
        weight_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'float': torch.float32}[self.mixed_precision]
        save_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'float': torch.float32}[self.save_precision]
        vae_dtype = torch.float32 if self.no_half_vae else weight_dtype
        return {'weight_dtype': weight_dtype, 'save_dtype': save_dtype, 'vae_dtype': vae_dtype}

    def _setup_model(self):
        for pi in range(self.accelerator.num_processes):
            if pi == self.accelerator.local_process_index:
                self.logger.print(f"loading model for process {self.accelerator.local_process_index}/{self.accelerator.num_processes}", disable=False)
                models = self.load_models()
                for key, model in models.items():
                    self.__dict__[key] = model
                gc.collect()
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

        if self.xformers:
            self.enable_xformers()

        self.vae.to(device=self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.get_vae_scale_factor()

    def load_models(self):
        models = {}
        for load_model in dir(self):
            if load_model.startswith("load_") and load_model.endswith('_model') and callable(getattr(self, load_model)):
                models.update(getattr(self, load_model)())
        return models

    def load_diffusion_model(self):
        models = {}
        tokenizer = model_utils.load_tokenizer(
            model_utils.TOKENIZER_PATH if not self.v2 else model_utils.V2_STABLE_DIFFUSION_PATH,
            subfolder=None if not self.v2 else 'tokenizer',
            cache_dir=self.tokenizer_cache_dir,
            max_token_length=self.max_token_length
        )
        models['tokenizer'] = tokenizer
        if os.path.isfile(self.pretrained_model_name_or_path):
            models_ = model_utils.load_models_from_stable_diffusion_checkpoint(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                v2=self.v2,
                nnet_class=self.nnet_class,
            )

        else:
            models_ = model_utils.load_models_from_stable_diffusion_diffusers_state(
                self.pretrained_model_name_or_path,
                device=self.device,
                dtype=self.weight_dtype,
                cache_dir=self.hf_cache_dir,
                nnet_class=self.nnet_class,
                max_retries=self.max_retries,
            )
        models.update(models_)
        if self.vae_model_name_or_path is not None:
            models['vae'] = model_utils.load_vae(self.vae_model_name_or_path, dtype=self.weight_dtype)
            self.logger.print(f"additional vae model loaded from {self.vae_model_name_or_path}")
        return models

    def enable_xformers(self):
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("Please install xformers to use the xformers model")
        self.nnet.set_use_memory_efficient_attention(True, False)
        if torch.__version__ >= "2.0.0":
            self.vae.set_use_memory_efficient_attention_xformers(True)

    def get_vae_scale_factor(self):
        return train_utils.VAE_SCALE_FACTOR

    def _setup_dataset(self):
        self.train_dataset = self.get_train_dataset()
        self.train_dataset.setup()
        if self.train_dataset.cache_latents:
            self.cache_image_latents()
        train_dataloader = self.get_train_dataloader()
        self.train_dataloader = train_dataloader

    def get_train_dataset(self):
        return self.dataset_class.from_config(
            self.config,
            self.accelerator,
            latents_dtype=self.weight_dtype,
        )

    def cache_image_latents(self):
        with torch.no_grad():
            self.train_dataset.cache_batch_latents(self.vae)
        self.vae.to('cpu')  # vae is not needed anymore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.accelerator.wait_for_everyone()

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=min(self.max_dataloader_n_workers, os.cpu_count() - 1),
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            persistent_workers=self.persistent_data_loader_workers,
        )

    def _setup_training(self):
        total_batch_size = self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
        num_train_epochs = self.num_train_epochs
        num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps / self.accelerator.num_processes)
        num_train_steps = num_train_epochs * num_steps_per_epoch
        self.total_batch_size, self.num_train_epochs, self.num_steps_per_epoch, self.num_train_steps = total_batch_size, num_train_epochs, num_steps_per_epoch, num_train_steps

    def _setup_params(self):
        training_models = []
        params_to_optimize = []

        num_train_params = 0
        for model_params_setter in dir(self):
            if model_params_setter.startswith("setup_") and model_params_setter.endswith('_params') and callable(getattr(self, model_params_setter)):
                try:
                    training_models_, params_to_optimize_ = getattr(self, model_params_setter)()
                except NotImplementedError:
                    continue
                for model, params in zip(training_models_, params_to_optimize_):
                    n_params = 0
                    training_models.append(model)
                    params_to_optimize.append(params)
                    for param in params['params']:
                        n_params += param.numel()
                    self.logger.print(f"{model.__class__.__name__}: {n_params} training parameters")
                    num_train_params += n_params

        if self.full_fp16:
            train_utils.patch_accelerator_for_fp16_training(self.accelerator)
        self.training_models, self.params_to_optimize = training_models, params_to_optimize
        self.num_train_params = num_train_params

    def setup_nnet_params(self):
        training_models = []
        params_to_optimize = []

        train_nnet = self.learning_rate > 0 or any([lr > 0 for lr in self.block_lr])
        if train_nnet:
            if self.gradient_checkpointing:
                self.nnet.enable_gradient_checkpointing()
            self.nnet.requires_grad_(True)
            training_models.append(self.nnet)
            if self.block_lr is None:
                params_to_optimize.append({"params": list(self.nnet.parameters()), "lr": self.learning_rate})
            else:
                # TODO: block_lr
                raise NotImplementedError
                # assert (
                #     isinstance(self.block_lr, list) and
                #     len(self.block_lr) == train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR
                # ), f"block_lr must have {sdxl_train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR} values"
                # params_to_optimize.extend(sdxl_train_utils.get_block_params_to_optimize(self.nnet, self.block_lr))
        else:
            self.nnet.requires_grad_(False)
        self.nnet.to(self.device, dtype=self.weight_dtype)

        self.nnet = self._prepare_one_model(self.nnet, train=train_nnet, transform_model_if_ddp=True)
        self.train_nnet = train_nnet
        self.learning_rate_nnet = self.block_lr or self.learning_rate

        return training_models, params_to_optimize

    def _setup_one_text_encoder_params(self, text_encoder, lr):
        training_models = []
        params_to_optimize = []

        lr = lr or self.learning_rate
        train_text_encoder = self.train_text_encoder and lr > 0
        if train_text_encoder:
            if self.gradient_checkpointing:
                text_encoder.gradient_checkpointing_enable()
            text_encoder.requires_grad_(True)
            text_encoder.train()
            training_models.append(text_encoder)
            params_to_optimize.append({"params": list(text_encoder.parameters()), "lr": lr})
        else:
            lr = 0
            text_encoder.to(self.weight_dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()

        text_encoder = self._prepare_one_model(text_encoder, train=train_text_encoder, transform_model_if_ddp=True)
        return training_models, params_to_optimize, text_encoder, train_text_encoder, lr

    def setup_text_encoder_params(self):
        (
            training_models,
            params_to_optimize,
            self.text_encoder,
            self.train_text_encoder,
            self.learning_rate_te
        ) = self._setup_one_text_encoder_params(
            self.text_encoder,
            self.learning_rate_te
        )
        return training_models, params_to_optimize

    def _prepare_one_model(self, model, train, transform_model_if_ddp=True):
        # Ensure weight dtype when full fp16/bf16 training
        if self.full_fp16:
            assert (
                self.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16'"
            self.logger.print(f"enable full fp16 training for {model.__class__.__name__}.")
            model.to(self.weight_dtype)
        elif self.full_bf16:
            assert (
                self.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16'"
            self.logger.print(f"enable full bf16 training for {model.__class__.__name__}.")
            model.to(self.weight_dtype)

        if train:
            model = self.accelerator.prepare(model)
            if transform_model_if_ddp:
                model, = train_utils.transform_models_if_DDP([model])
        model.to(self.device)
        return model

    def _setup_optims(self):
        optimizer = train_utils.get_optimizer(self.config, self.params_to_optimize)
        lr_scheduler = train_utils.get_scheduler_fix(self.config, optimizer, self.num_train_steps)
        self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            optimizer, lr_scheduler, self.train_dataloader
        )

    def _setup_noise_scheduler(self):
        self.noise_scheduler = self.get_noise_scheduler()

    def get_noise_scheduler(self):
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        if self.prediction_type is not None:  # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=self.prediction_type)
        train_utils.prepare_scheduler_for_custom_training(noise_scheduler, self.device)
        if self.zero_terminal_snr:
            advanced_train_utils.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        return noise_scheduler

    def _setup_loss_recorder(self):
        self.loss_recorder = self.get_loss_recorder()

    def get_loss_recorder(self):
        return train_utils.LossRecorder(
            gamma=self.loss_recorder_kwargs.gamma, max_window=min(self.num_steps_per_epoch, 10000)
        )

    def _setup_train_state(self):
        self.train_state = self.get_train_state()

    def get_train_state(self):
        return self.train_state_class.from_config(
            self.config,
            self.accelerator,
            pipeline_class=self.pipeline_class,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_dataloader,
            save_dtype=self.save_dtype,
            nnet=self.nnet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
        )

    def _print_start_training_message(self):
        self.logger.print(log_utils.green(f"==================== START TRAINING ===================="))
        self.logger.print(f"  num train steps: {log_utils.yellow(self.num_train_epochs)} x {log_utils.yellow(self.num_steps_per_epoch)} = {log_utils.yellow(self.num_train_steps)}")
        self.logger.print(f"  number of trainable parameters: {self.num_train_params} = {self.num_train_params / 1e9:.3f}B")
        self.logger.print(
            f"  total batch size: {log_utils.yellow(self.total_batch_size)} = {self.batch_size} (batch size) x {self.gradient_accumulation_steps} (gradient accumulation steps) x {self.accelerator.num_processes} (num processes)"
        )
        self.logger.print(f"  mixed precision: {self.mixed_precision} | weight-dtype: {self.weight_dtype} | save-dtype: {self.save_dtype}")
        self.logger.print(f"  optimizer: {self.optimizer_type} | timestep sampler: {self.timestep_sampler_type}")
        self.logger.print(f"  device: {log_utils.yellow(self.device)}")

    def train(self):
        self._print_start_training_message()
        self.pbar = self.train_state.pbar()
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers('finetune', init_kwargs={})

        try:
            self.train_loop()
        except KeyboardInterrupt:
            save_on_train_end = self.accelerator.is_main_process and self.save_on_keyboard_interrupt
            self.logger.print("KeyboardInterrupted.")
        except Exception as e:
            import traceback
            save_on_train_end = self.accelerator.is_main_process and self.save_on_exception
            self.logger.print("Exception:", e)
            traceback.print_exc()
        else:
            save_on_train_end = self.accelerator.is_main_process and self.save_on_train_end

        self.pbar.close()
        self.accelerator.wait_for_everyone()
        if save_on_train_end:
            self.logger.print(f"saving on train end...")
            self.train_state.save(on_train_end=True)
        self.accelerator.end_training()
        self.logger.print(log_utils.green(f"training finished at process {self.accelerator.local_process_index+1}/{self.accelerator.num_processes}"), disable=False)
        del self.accelerator

    def get_loss(self, model_pred, target, timesteps):
        if self.min_snr_gamma or self.debiased_estimation_loss:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            if self.min_snr_gamma:
                loss = advanced_train_utils.apply_snr_weight(loss, timesteps, self.noise_scheduler, self.min_snr_gamma, self.prediction_type)
            if self.debiased_estimation_loss:
                loss = advanced_train_utils.apply_debiased_estimation(loss, timesteps, self.noise_scheduler)

            loss = loss.mean()
        else:
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss):
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return loss

    def get_noise(self, latents):
        noise = torch.randn_like(latents, device=latents.device)
        if self.noise_offset:
            noise = advanced_train_utils.apply_noise_offset(latents, noise, self.noise_offset, self.adaptive_noise_scale)
        if self.multires_noise_iterations:
            noise = advanced_train_utils.pyramid_noise_like(
                noise, latents.device, self.multires_noise_iterations, self.multires_noise_discount
            )
        return noise

    def get_timesteps(self, latents):
        b_size = latents.shape[0]
        min_timestep = 0 if self.min_timestep is None else self.min_timestep
        max_timestep = self.noise_scheduler.config.num_train_timesteps if self.max_timestep is None else self.max_timestep

        if self.timestep_sampler_type == "uniform":
            timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)  # timestep is in [min_timestep, max_timestep)
            timesteps = timesteps.long()
        elif self.timestep_sampler_type == "logit_normal":  # Rectified Flow from SD3 paper (partial implementation)
            timestep_sampler_kwargs = self.timestep_sampler_kwargs
            m = timestep_sampler_kwargs.get('loc', 0) or timestep_sampler_kwargs.get('mean', 0) or timestep_sampler_kwargs.get('m', 0) or timestep_sampler_kwargs.get('mu', 0)
            s = timestep_sampler_kwargs.get('scale', 1) or timestep_sampler_kwargs.get('std', 1) or timestep_sampler_kwargs.get('s', 1) or timestep_sampler_kwargs.get('sigma', 1)
            timesteps = advanced_train_utils.logit_normal(mu=m, sigma=s, shape=(b_size,), device=latents.device)  # sample from logistic normal distribution
            timesteps = timesteps * (max_timestep - min_timestep) + min_timestep  # scale to [min_timestep, max_timestep)
            timesteps = timesteps.long()

        return timesteps

    def get_noisy_latents(self, latents, noise, timesteps):
        if self.ip_noise_gamma:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise + self.ip_noise_gamma * torch.randn_like(latents), timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
                if torch.any(torch.isnan(latents)):
                    self.pbar.write("NaN found in latents, replacing with zeros")
                    latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        latents *= self.vae_scale_factor

        input_ids = torch.stack([train_utils.get_input_ids(caption, self.tokenizer, max_token_length=self.max_token_length) for caption in batch['captions']], dim=0)
        with torch.set_grad_enabled(self.train_text_encoder):
            input_ids = input_ids.to(self.device)
            encoder_hidden_states = train_utils.get_hidden_states(
                input_ids, self.tokenizer, self.text_encoder, weight_dtype=None if not self.full_fp16 else self.weight_dtype,
                v2=self.v2, clip_skip=self.clip_skip, max_token_length=self.max_token_length,
            )

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(self.weight_dtype)

        with self.accelerator.autocast():
            model_pred = self.nnet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps)
        return loss

    def train_loop(self):
        while self.train_state.epoch < self.num_train_epochs:
            if self.accelerator.is_main_process:
                self.pbar.write(f"epoch: {self.train_state.epoch}/{self.num_train_epochs}")
            for m in self.training_models:
                m.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(*self.training_models):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in self.training_models:
                            params_to_clip.extend(m.parameters())
                        self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.pbar.update(1)
                    self.train_state.step()
                    self.train_state.save(on_step_end=True)
                    self.train_state.sample(on_step_end=True)

                # loggings
                step_loss: float = loss.detach().item()
                self.loss_recorder.add(loss=step_loss)
                avr_loss: float = self.loss_recorder.moving_average(window=self.loss_recorder_kwargs.stride)
                ema_loss: float = self.loss_recorder.ema
                logs = {"loss/step": step_loss, 'loss_avr/step': avr_loss, 'loss_ema/step': ema_loss}
                self.accelerator.log(logs, step=self.train_state.global_step)
                pbar_logs = {
                    'lr': self.lr_scheduler.get_last_lr()[0],
                    'epoch': self.train_state.epoch,
                    'global_step': self.train_state.global_step,
                    'next': len(self.train_dataloader) - step - 1,
                    'step_loss': step_loss,
                    'avr_loss': avr_loss,
                    'ema_loss': ema_loss,
                }
                self.pbar.set_postfix(pbar_logs)

            # end of epoch
            logs = {"loss/epoch": self.loss_recorder.moving_average(window=self.num_steps_per_epoch)}
            self.accelerator.log(logs, step=self.train_state.epoch)
            self.accelerator.wait_for_everyone()
            self.train_state.save(on_epoch_end=True)
            self.train_state.sample(on_epoch_end=True)
            if self.train_state.global_step >= self.num_train_steps:
                break
