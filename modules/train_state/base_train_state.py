import torch
import os
import gc
import shutil
from typing import Dict
from ml_collections import ConfigDict
from argparse import Namespace
from accelerate import Accelerator
from typing import Any, Literal, List
from waifuset import logging
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import EMAModel
from ..datasets.base_dataset import BaseDataset
from ..utils import class_utils


class BaseTrainState(class_utils.SubModuleMixin):
    r"""
    Train state is a sub-module of the trainer that manages the training state, including saving and evaluation.
    """

    trainer: Any
    accelerator: Accelerator
    logger: logging.ConsoleLogger
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    train_dataset: BaseDataset
    valid_dataset: BaseDataset
    train_dataloader: torch.utils.data.DataLoader
    valid_dataloader: torch.utils.data.DataLoader
    models: dict
    train_dataset: BaseDataset
    save_dtype: torch.dtype

    save_model: bool = True
    save_train_state: bool = True
    save_every_n_steps: int = None
    save_every_n_epochs: int = 1
    save_on_train_start: bool = False
    save_on_train_end: bool = True
    save_on_steps: List[int] = []
    save_on_keyboard_interrupt: bool = False
    save_on_exception: bool = False
    save_max_n_models: int = None
    save_max_n_train_states: int = None
    save_max_n_ema_models: int = None
    save_as_format: Literal['torch', 'safetensors'] = 'safetensors'

    eval_every_n_steps: int = None
    eval_every_n_epochs: int = 1
    eval_on_train_start: bool = False
    eval_on_train_end: bool = False
    eval_on_steps: List[int] = []
    eval_on_keyboard_interrupt: bool = False
    eval_on_exception: bool = False

    # inherited from trainer's config
    output_dir: str
    output_subdir: ConfigDict
    output_name: ConfigDict
    batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int

    save_precision: Literal['fp16', 'bf16', 'float'] = 'fp16'
    resume_from: str = None

    use_ema: bool = False
    save_ema: bool = True
    ema_models: Dict[str, EMAModel] = {}

    def setup(self):
        self.logger = logging.get_logger('train_state', disable=not self.accelerator.is_local_main_process)
        self.check_config()

        self.output_model_dir = os.path.join(self.output_dir, self.output_subdir.models)
        self.output_train_state_dir = os.path.join(self.output_dir, self.output_subdir.train_state)

        output_name_default = os.path.basename(self.output_dir)
        self.output_name = dict(
            models=self.output_name.models or output_name_default,
            # samples=self.output_name.samples or output_name_default,
            train_state=self.output_name.train_state or output_name_default,
        )

        self.global_step = 0

        self.save_model_history = []
        self.save_ema_model_history = []
        self.save_train_state_history = []

        self.logger.info(f"Event triggers: {', '.join([logging.yellow(trigger.__name__) for trigger in self.get_event_triggers()])}")

    def check_config(self):
        if self.save_max_n_models is not None and self.save_max_n_models < 1:
            raise ValueError(f"save_max_n_models must be greater than 0, got {self.save_max_n_models}")
        if self.save_max_n_train_states is not None and self.save_max_n_train_states < 1:
            raise ValueError(f"save_max_n_train_states must be greater than 0, got {self.save_max_n_train_states}")

    def step(self):
        self.global_step += 1

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch

    def save(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_save = False
        do_save |= bool(on_step_end and self.global_step and self.save_every_n_steps and self.global_step % self.save_every_n_steps == 0)
        do_save |= bool(on_epoch_end and self.epoch and self.save_every_n_epochs and self.epoch % self.save_every_n_epochs == 0)
        do_save |= bool(on_train_end)
        do_save |= bool(self.save_on_train_start and self.global_step == 0)
        do_save |= bool(self.save_on_steps and self.global_step in self.save_on_steps)
        do_save &= bool(self.save_model or self.save_train_state)
        if do_save:
            self.do_save(on_step_end=on_step_end, on_epoch_end=on_epoch_end, on_train_end=on_train_end)

    def do_save(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.save_model:
                try:
                    save_model_paths = self.save_models_to_disk()
                    self.save_model_history.append(save_model_paths)
                    if self.use_ema and self.save_ema:
                        save_ema_paths = self.save_ema_models_to_disk()
                        self.save_ema_model_history.append(save_ema_paths)
                except Exception as e:
                    import traceback
                    self.logger.print(logging.red("exception when saving model:", e))
                    traceback.print_exc()
                    pass
            if self.save_train_state:
                try:
                    save_train_state_path = self.save_train_state_to_disk()
                    self.save_train_state_history.append(save_train_state_path)
                except Exception as e:
                    import traceback
                    self.logger.print(logging.red("exception when saving train state:", e))
                    traceback.print_exc()
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        self.accelerator.wait_for_everyone()

        if self.save_max_n_models:
            for save_paths in self.save_model_history[:-self.save_max_n_models]:
                for save_path in save_paths:
                    if save_path is not None and os.path.exists(save_path):
                        try:
                            os.remove(save_path) if os.path.isfile(save_path) else shutil.rmtree(save_path)
                        except Exception as e:
                            import traceback
                            self.logger.print(logging.red(f"exception when removing model: {save_path}", e))
                            traceback.print_exc()
                            pass

        if self.save_max_n_ema_models:
            for save_paths in self.save_ema_model_history[:-self.save_max_n_ema_models]:
                for save_path in save_paths:
                    if save_path is not None and os.path.exists(save_path):
                        try:
                            os.remove(save_path) if os.path.isfile(save_path) else shutil.rmtree(save_path)
                        except Exception as e:
                            import traceback
                            self.logger.print(logging.red(f"exception when removing ema model: {save_path}", e))
                            traceback.print_exc()
                            pass

        if self.save_max_n_train_states:
            for save_path in self.save_train_state_history[:-self.save_max_n_train_states]:
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path) if os.path.isfile(save_path) else shutil.rmtree(save_path)
                    except Exception as e:
                        import traceback
                        self.logger.print(logging.red(f"exception when removing train state: {save_path}", e))
                        traceback.print_exc()
                        pass

    def save_models_to_disk(self) -> List[str]:
        self.logger.print(f"saving model at epoch {self.epoch}, step {self.global_step}...")
        with self.logger.timer("save models"):
            save_paths = []
            for saver in dir(self):
                if saver.startswith('save_') and saver.endswith('_model') and callable(saver := getattr(self, saver)):
                    save_path = saver()
                    save_paths.append(save_path)
        return save_paths

    def save_ema_models_to_disk(self) -> str:
        self.logger.print(f"saving ema model at epoch {self.epoch}, step {self.global_step}...")
        save_paths = []
        with self.logger.timer(f"save ema models"):
            for ema_name, ema_model in self.ema_models.items():
                ema_model = self.unwrap_model(ema_model)
                save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_{ema_name}_ema_ep{self.epoch}_step{self.global_step}")
                ema_model.save_pretrained(save_path)
                save_paths.append(save_path)
        return save_paths

    def save_train_state_to_disk(self) -> str:
        self.logger.print(f"saving train state at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_train_state_dir, f"{self.output_name['train_state']}_train-state_ep{self.epoch}_step{self.global_step}")
        with self.logger.timer(f"save train state to {save_path}"):
            self.accelerator.save_state(save_path)
        return save_path

    def get_save_extension(self):
        if self.save_as_format == 'torch':
            return '.pt'
        elif self.save_as_format == 'safetensors':
            return '.safetensors'
        else:
            raise ValueError(f"save_as_format must be 'torch' or 'safetensors', but got {self.save_as_format}")

    def unwrap_model(self, model):
        try:
            model = self.accelerator.unwrap_model(model)
        except:
            pass
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def resume(self):
        if self.resume_from:
            self.accelerator.load_state(self.resume_from)
            self.global_step = self.accelerator.step
            self.logger.print(f"train state loaded from: `{logging.yellow(self.resume_from)}`")

    def pbar(self):
        from tqdm import tqdm
        return tqdm(total=self.num_train_steps, initial=self.global_step, desc='steps', disable=not self.accelerator.is_local_main_process)

    def eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_eval = False
        do_eval |= bool(on_step_end and self.eval_every_n_steps and self.global_step % self.eval_every_n_steps == 0)
        do_eval |= bool(on_epoch_end and self.eval_every_n_epochs and self.epoch % self.eval_every_n_epochs == 0)
        do_eval |= bool(self.eval_on_train_end)
        do_eval |= bool(self.eval_on_steps and self.global_step in self.eval_on_steps)
        do_eval |= bool(on_train_end)
        on_train_start = bool(self.eval_on_train_start and self.global_step == 0)
        do_eval |= on_train_start
        if do_eval:
            self.do_eval(on_step_end=on_step_end, on_epoch_end=on_epoch_end, on_train_end=on_train_end, on_train_start=on_train_start)

    def do_eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False, on_train_start=False):
        pass

    def get_event_triggers(self):
        return [getattr(self, func) for func in dir(self) if func.startswith("trigger_") and func.endswith("_event") and callable(getattr(self, func))]

    def trigger_events(self):
        for trigger in self.get_event_triggers():
            trigger()
