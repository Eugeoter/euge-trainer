import torch
import math
import os
import gc
from argparse import Namespace
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from typing import Literal
from ..utils import log_utils, class_utils, model_utils, eval_utils

logger = log_utils.get_logger("train")


class TrainState(class_utils.FromConfigMixin):
    accelerator: Accelerator
    logger: log_utils.ConsoleLogger
    pipeline_class: type
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    train_dataloader: torch.utils.data.DataLoader
    models: dict
    save_dtype: torch.dtype

    batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int

    save_model: bool = True
    save_train_state: bool = True
    save_every_n_steps: int = None
    save_every_n_epochs: int = 1
    save_on_train_end: bool = True
    save_on_keyboard_interrupt: bool = False
    save_on_exception: bool = False

    sample_benchmark: str = None
    sample_every_n_steps: int = None
    sample_every_n_epochs: int = 1
    sample_at_first: bool = False
    sample_sampler: str = 'euler_a'
    sample_params = class_utils.cfg(
        prompt="1girl, solo, cowboy shot, white background, smile, looking at viewer, serafuku, pleated skirt",
        negative_prompt="abstract, bad anatomy, clumsy pose, signature",
        steps=28,
        batch_size=1,
        batch_count=4,
        scale=7.5,
        seed=42,
        width=832,
        height=1216,
        save_latents=False,
    )

    output_dir: str
    output_subdir = class_utils.cfg(
        models='models',
        train_state='train_state',
        samples='samples',
        logs='logs',
    )
    output_name = class_utils.cfg(
        models=None,
        train_state=None,
    )
    save_precision: Literal['fp16', 'bf16', 'float'] = 'fp16'
    resume_from: str = None

    @classmethod
    def from_config(
        cls,
        config,
        accelerator,
        pipeline_class,
        optimizer,
        lr_scheduler,
        train_dataloader,
        save_dtype,
        **kwargs
    ):
        models = {}
        for key, val in kwargs.items():
            if isinstance(val, torch.nn.Module) or (isinstance(val, (tuple, list)) and any(isinstance(v, torch.nn.Module) for v in val)):
                models[key] = val
        models = Namespace(**models)
        logger = log_utils.get_logger('train_state', disable=not accelerator.is_local_main_process)
        return super().from_config(
            config,
            accelerator=accelerator,
            logger=logger,
            pipeline_class=pipeline_class,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            models=models,
            save_dtype=save_dtype,
            **kwargs,
        )

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.setup()

    def setup(self):
        self.save_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'float': torch.float32}[self.save_precision]

        self.total_batch_size = self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
        self.num_train_epochs = self.num_train_epochs
        self.num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps / self.accelerator.num_processes)
        self.num_train_steps = self.num_train_epochs * self.num_steps_per_epoch

        self.output_model_dir = os.path.join(self.output_dir, self.output_subdir.models)
        self.output_train_state_dir = os.path.join(self.output_dir, self.output_subdir.train_state)

        output_name_default = os.path.basename(self.output_dir)
        self.output_name = dict(
            models=self.output_name.models or output_name_default,
            # samples=self.output_name.samples or output_name_default,
            train_state=self.output_name.train_state or output_name_default,
        )

        self.global_step = 0

    def step(self):
        self.global_step += 1

    @ property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch

    def save_models_to_disk(self):
        self.logger.print(f"saving model at epoch {self.epoch}, step {self.global_step}...")
        for saver in dir(self):
            if saver.startswith('save_') and saver.endswith('_model') and callable(saver := getattr(self, saver)):
                saver()

    def save_diffusion_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        model_utils.save_stable_diffusion_checkpoint(
            output_file=save_path,
            unet=self.accelerator.unwrap_model(self.models['unet']),
            text_encoder=self.accelerator.unwrap_model(self.models['text_encoder']),
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_path=None,
            vae=self.models['vae'],
            save_dtype=self.save_dtype,
            metadata=None,
            v2=self.v2,
        )
        self.logger.print(f"diffusion model saved to: `{log_utils.yellow(save_path)}`")

    def save_train_state_to_disk(self):
        self.logger.print(f"saving train state at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_train_state_dir, f"{self.output_name['train_state']}_train-state_ep{self.epoch}_step{self.global_step}")
        self.accelerator.save_state(save_path)
        self.logger.print(f"train state saved to: `{log_utils.yellow(save_path)}`")

    def save(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_save = False
        do_save |= bool(on_step_end and self.global_step and self.save_every_n_steps and self.global_step % self.save_every_n_steps == 0)
        do_save |= bool(on_epoch_end and self.epoch and self.save_every_n_epochs and self.epoch % self.save_every_n_epochs == 0)
        do_save |= bool(on_train_end)
        do_save &= bool(self.save_model or self.save_train_state)
        if do_save:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.save_model:
                    try:
                        self.save_models_to_disk()
                    except Exception as e:
                        import traceback
                        self.logger.print(log_utils.red("exception when saving model:", e))
                        traceback.print_exc()
                        pass
                if self.save_train_state:
                    try:
                        self.save_train_state_to_disk()
                    except Exception as e:
                        import traceback
                        self.logger.print(log_utils.red("exception when saving train state:", e))
                        traceback.print_exc()
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            self.accelerator.wait_for_everyone()

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.models.nnet),
            text_encoder=self.unwrap_model(self.models.text_encoder),
            tokenizer=self.tokenizer,
            vae=self.unwrap_model(self.models.vae),
            scheduler=eval_utils.get_sampler(self.sample_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=self.clip_skip,
        )

    def unwrap_model(self, model):
        try:
            model = self.accelerator.unwrap_model(model)
        except:
            pass
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def sample(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_sample = False
        do_sample |= bool(on_step_end and self.sample_every_n_steps and self.global_step % self.sample_every_n_steps == 0)
        do_sample |= bool(on_epoch_end and self.sample_every_n_epochs and self.epoch % self.sample_every_n_epochs == 0)
        do_sample |= bool(on_train_end)
        do_sample |= bool(self.sample_at_first and self.global_step == 0)
        if do_sample:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                try:
                    sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"ep{self.epoch}_step{self.global_step}")
                    eval_utils.sample_during_train(
                        pipeline=self.get_pipeline(),
                        sample_dir=sample_dir,
                        benchmark_file=self.sample_benchmark,
                        default_params=self.sample_params,
                        accelerator=self.accelerator,
                        epoch=self.epoch if on_epoch_end else None,
                        steps=self.global_step,
                        device=self.accelerator.device,
                    )
                except Exception as e:
                    import traceback
                    self.logger.print(log_utils.red("exception when sample images:", e))
                    traceback.print_exc()
                    pass
            self.accelerator.wait_for_everyone()

    def resume(self):
        if self.resume_from:
            self.accelerator.load_state(self.resume_from)
            self.global_step = self.accelerator.step
            self.logger.print(f"train state loaded from: `{log_utils.yellow(self.resume_from)}`")

    def pbar(self):
        from tqdm import tqdm
        return tqdm(total=self.num_train_steps, initial=self.global_step, desc='steps', disable=not self.accelerator.is_local_main_process)
