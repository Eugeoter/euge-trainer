import torch
import os
import gc
import math
import time
import json
import contextlib
import random
import pickle
import numpy as np
import torch.distributed as dist
from accelerate import Accelerator
from typing import Optional, List, Union, Literal, Any, Dict, Callable
from waifuset import logging
from waifuset import DictDataset
from diffusers.training_utils import EMAModel
from ..utils import class_utils, debug_utils, deepspeed_utils, train_utils, device_utils
from ..datasets.base_dataset import BaseDataset
from ..train_state.base_train_state import BaseTrainState


class BaseTrainer(class_utils.FromConfigMixin):
    dataset_class = BaseDataset
    train_state_class = BaseTrainState

    train_dataset: dataset_class
    train_dataloader: torch.utils.data.DataLoader
    train_state: BaseTrainState

    dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    dataset_merge_mode: Literal['union', 'intersection'] = 'union'
    dataset_cache_path: Optional[str] = None
    dataset_full_cache_path: Optional[str] = None
    valid_dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]] = None
    valid_dataset_merge_mode: Literal['union', 'intersection'] = 'union'
    valid_dataset_cache_path: Optional[str] = None
    valid_dataset_full_cache_path: Optional[str] = None
    dataset_splitter: Optional[Callable[[Dict[str, Any]], Literal['train', 'validation']]] = None
    valid_dataset: dataset_class
    valid_dataloader: torch.utils.data.DataLoader

    loss_weight_getter: Callable[[Dict[str, Any]], float] = None

    output_dir: str
    output_subdir = class_utils.cfg(
        models='models',
        train_state='train_state',
        logs='logs',
    )
    output_name = class_utils.cfg(
        models=None,
        train_state=None,
    )

    use_tensorboard: bool = False

    use_wandb: bool = False
    wandb_project: str = None
    wandb_entity: str = None
    wandb_name: str = None
    wandb_token: str = None
    wandb_resume: Literal['allow', 'must', 'never', 'auto'] = 'never'
    wandb_id: str = None
    wandb_tags: List[str] = []
    wandb_save_code: bool = False

    note: Optional[str] = None  # human-readable notes for the run
    seed: int = None

    hf_cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    max_retries: int = None  # = infinite retries

    learning_rate: float
    lr_scheduler_type: str = 'constant_with_warmup'
    lr_scheduler_kwargs: Dict[str, Any] = {}
    lr_warmup_steps: int = 0
    lr_scheduler_power: float = 1.0
    lr_scheduler_num_cycles: int = 1
    lr_scheduler_kwargs: dict = class_utils.cfg()

    batch_size: int
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_grad_norm: float = 0.5
    mixed_precision: Literal['fp16', 'bf16', 'no'] = 'fp16'
    save_precision: Literal['fp16', 'bf16', 'no'] = 'fp16'
    full_fp16: bool = False
    full_bf16: bool = False

    weight_dtype: torch.dtype
    save_dtype: torch.dtype

    ema_models: Dict[str, EMAModel] = {}
    use_ema: bool = False
    save_ema: bool = True

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

    optimizer_type: str
    optimizer_kwargs: dict = {}

    lr_scheduler_type: str
    lr_scheduler_kwargs: dict = {}

    cpu: bool = False

    use_deepspeed: bool = False
    deepspeed_remote_device: str = 'none'
    zero_stage: int = 2
    cpu_offloading: bool = True
    offload_optimizer_device: Literal[None, "cpu", "nvme"] = None
    offload_optimizer_nvme_path: str = None
    offload_param_device: Literal[None, "cpu", "nvme"] = None
    offload_param_nvme_path: str = None
    zero3_init_flag: bool = False
    zero3_save_16bit_model: bool = False
    fp16_master_weights_and_gradients: bool = False
    zero3_init_flag: bool = False
    zero3_save_16bit_model: bool = False

    def check_config(self):
        r"""
        Check the validity of the configuration. Raise ValueError if the configuration is invalid. Warn if there are inconsistencies in the configuration.
        """
        if self.learning_rate is not None and not 0 <= self.learning_rate <= 1:
            raise ValueError(f"`learning_rate` must be in (0, 1), but got {self.learning_rate}")
        if self.lr_warmup_steps is not None and self.lr_warmup_steps < 0:
            raise ValueError(f"`lr_warmup_steps` must be non-negative, but got {self.lr_warmup_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, but got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"`gradient_accumulation_steps` must be positive, but got {self.gradient_accumulation_steps}")
        if self.mixed_precision not in ['fp16', 'bf16', 'no']:
            raise ValueError(f"`mixed_precision` must be one of ['fp16', 'bf16', 'no'], but got {self.mixed_precision}")

        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"`note` must be a string, but got {self.note}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"`seed` must be an integer, but got type {type(self.seed)}")
        if self.hf_cache_dir is not None and not isinstance(self.hf_cache_dir, str):
            raise ValueError(f"`hf_cache_dir` must be a string, but got type {type(self.hf_cache_dir)}")
        if self.hf_token is not None and not isinstance(self.hf_token, str):
            raise ValueError(f"`hf_token` must be a string, but got type {type(self.hf_token)}")
        if self.max_retries is not None:
            if not isinstance(self.max_retries, int):
                raise ValueError(f"`max_retries` must be an integer, but got type {type(self.max_retries)}")
            if self.max_retries <= 0:
                raise ValueError(f"`max_retries` must be positive, but got {self.max_retries}")

        if self.dataset_splitter is not None and self.valid_dataset_source is not None:
            raise ValueError(f"`dataset_splitter` and `valid_dataset_source` cannot be used together")

        if self.dataset_splitter is not None and self.dataset_full_cache_path is not None:
            self.logger.warning(f"`dataset_splitter` is ignored when `dataset_full_cache_path` is used")
        if self.valid_dataset_source is not None and self.valid_dataset_full_cache_path is not None:
            self.logger.warning(f"`valid_dataset_source` is ignored when `valid_dataset_full_cache_path` is used")

        if self.dataset_cache_path is not None and self.dataset_full_cache_path is not None:
            self.logger.warning(f"`dataset_cache_path` is ignored for loading when `dataset_full_cache_path` is used")
        if self.valid_dataset_cache_path is not None and self.valid_dataset_full_cache_path is not None:
            self.logger.warning(f"`valid_dataset_cache_path` is ignored for loading when `valid_dataset_full_cache_path` is used")

    def setup(self):
        r"""
        Setup the training environment in a specific order.
        """
        self._setup_basic()
        setups = self.get_setups()
        with self.logger.timer("setup training"):
            for setup in setups:
                setup()

    def get_setups(self):
        r"""
        Return a list of setup functions in the order they should be called.

        Subclasses should override this method to define the order of setup functions.
        """
        return [
            self._setup_dtype,
            self._setup_model,
            self._setup_dataset,
            self._setup_training,
            self._setup_params,
            self._setup_optims,
            self._setup_loss_recorder,
            self._setup_train_state,
        ]

    def _setup_basic(self):
        r"""
        Setup seed, accelerator/deepspeed, logger and device, check config, ignore warnings, save config and note.
        """
        if self.use_deepspeed:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            self.device = rank % torch.cuda.device_count()
            torch.cuda.set_device(self.device)
        else:
            self.accelerator = self.get_accelerator()
            self.device = self.accelerator.device
        self.logger = self.get_logger()
        self.check_config()
        if self.seed is not None:
            self.set_seed()
        if self.ignore_warnings:
            debug_utils.ignore_warnings()
        self.save_config(os.path.join(self.output_dir, 'config.json'))
        if self.note is not None:
            self.save_note(os.path.join(self.output_dir, 'note.txt'))

    def _setup_dtype(self):
        r"""
        Setup `weight_dtype` and `save_dtype`.
        """
        dtypes = self.get_dtypes()
        for key, dtype in dtypes.items():
            self.__dict__[key] = dtype

    def _setup_model(self):
        r"""
        Setup all models.
        """
        for pi in range(self.accelerator.num_processes):
            if pi == self.accelerator.local_process_index:
                self.logger.info(f"loading model for process {self.accelerator.local_process_index}/{self.accelerator.num_processes}", disable=False)
                models = self.load_models()
                for key, model in models.items():
                    self.__dict__[key] = model
                gc.collect()
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
        self.models = models

    def _setup_dataset(self):
        r"""
        Setup train dataset and dataloader.
        """
        self.logger.info(f"Setting up datasets...")

        # Load full datasets from cache
        self.train_dataset, self.valid_dataset = None, None
        has_train_dataset_setup, has_valid_dataset_setup = False, False
        if self.dataset_full_cache_path is not None and os.path.exists(self.dataset_full_cache_path):
            self.logger.info(f"Loading full train dataset from cache `{logging.yellow(self.dataset_full_cache_path)}`")
            with self.logger.timer("Load full train dataset"):
                with open(self.dataset_full_cache_path, 'rb') as f:
                    self.train_dataset = pickle.load(f)
            has_train_dataset_setup = True
        if self.valid_dataset_full_cache_path is not None and os.path.exists(self.valid_dataset_full_cache_path):
            self.logger.info(f"Loading full validation dataset from cache `{logging.yellow(self.valid_dataset_full_cache_path)}`")
            with self.logger.timer("Load full validation dataset"):
                with open(self.valid_dataset_full_cache_path, 'rb') as f:
                    self.valid_dataset = pickle.load(f)
            has_valid_dataset_setup = True

        # Load datasets from cache
        train_dataset_source, valid_dataset_source = None, None
        if self.train_dataset is None and self.dataset_cache_path is not None and os.path.exists(self.dataset_cache_path):
            self.logger.info(f"Loading train dataset from cache `{logging.yellow(self.dataset_cache_path)}`")
            with self.logger.timer("Load train dataset"):
                with open(self.dataset_cache_path, 'rb') as f:
                    train_dataset_source = pickle.load(f)
        if self.valid_dataset is None and self.valid_dataset_cache_path is not None and os.path.exists(self.valid_dataset_cache_path):
            self.logger.info(f"Loading validation dataset from cache `{logging.yellow(self.valid_dataset_cache_path)}`")
            with self.logger.timer("Load validation dataset"):
                with open(self.valid_dataset_cache_path, 'rb') as f:
                    valid_dataset_source = pickle.load(f)

        if self.dataset_splitter is not None and self.train_dataset is None and self.valid_dataset is None and valid_dataset_source is None:
            self.logger.info(f"Splitting dataset with splitter `{logging.yellow(self.dataset_splitter.__name__)}`")
            if train_dataset_source is None:
                dataset = self.get_dataset(setup=False)
            else:
                dataset = self.get_dataset(setup=False, dataset_source=train_dataset_source)
            dataset._setup_basic()
            dataset = dataset.load_dataset()

            train_data_dict = {}
            valid_data_dict = {}
            for img_key, img_md in self.logger.tqdm(dataset.items(), desc='split dataset', total=len(dataset)):
                split = self.dataset_splitter(img_md)
                if split == 'train':
                    train_data_dict[img_key] = img_md
                elif split == 'validation':
                    valid_data_dict[img_key] = img_md
                else:
                    raise ValueError(f"invalid split `{split}`, must be 'train' or 'validation'")
            train_dataset_source = DictDataset.from_dict(train_data_dict)
            valid_dataset_source = DictDataset.from_dict(valid_data_dict)
        else:
            if train_dataset_source is None:
                train_dataset_source = self.dataset_source
            if valid_dataset_source is None:
                valid_dataset_source = self.valid_dataset_source

        if self.train_dataset is None:
            self.train_dataset = self.get_dataset(
                dataset_source=train_dataset_source,
                dataset_merge_mode=self.dataset_merge_mode,
                setup=False,
            )
            self.train_dataset.dataset = self.train_dataset.load_dataset()
        if self.valid_dataset is None and valid_dataset_source is not None:
            self.valid_dataset = self.get_dataset(
                dataset_source=valid_dataset_source,
                dataset_merge_mode=self.valid_dataset_merge_mode,
                record_dataset_info=False,
                setup=False,
            )
            self.valid_dataset.dataset = self.valid_dataset.load_dataset()

        if self.dataset_cache_path is not None and not os.path.exists(self.dataset_cache_path):
            self.logger.info(f"Caching train dataset to `{logging.yellow(self.dataset_cache_path)}`")
            with self.logger.timer("Cache train dataset"):
                with open(self.dataset_cache_path, 'wb') as f:
                    pickle.dump(self.train_dataset.dataset, f)
        if self.valid_dataset_cache_path is not None and not os.path.exists(self.valid_dataset_cache_path):
            self.logger.info(f"Caching validation dataset to `{logging.yellow(self.valid_dataset_cache_path)}`")
            with self.logger.timer("Cache validation dataset"):
                with open(self.valid_dataset_cache_path, 'wb') as f:
                    pickle.dump(self.valid_dataset.dataset, f)

        if not has_train_dataset_setup:
            self.train_dataset.setup()
        if self.valid_dataset is not None and not has_valid_dataset_setup:
            self.valid_dataset.setup()

        if self.dataset_full_cache_path is not None and not os.path.exists(self.dataset_full_cache_path):
            self.logger.info(f"Caching full train dataset to `{self.dataset_full_cache_path}`")
            with self.logger.timer("Cache full train dataset"):
                with open(self.dataset_full_cache_path, 'wb') as f:
                    pickle.dump(self.train_dataset, f)
        if self.valid_dataset and self.valid_dataset_full_cache_path is not None and not os.path.exists(self.valid_dataset_full_cache_path):
            self.logger.info(f"Caching full validation dataset to `{self.valid_dataset_full_cache_path}`")
            with self.logger.timer("Cache full validation dataset"):
                with open(self.valid_dataset_full_cache_path, 'wb') as f:
                    pickle.dump(self.valid_dataset, f)

        self.logger.info(f"Setting up data loaders...")
        self.train_dataloader = self.get_dataloader(self.train_dataset)
        self.valid_dataloader = self.get_dataloader(self.valid_dataset, shuffle=False) if self.valid_dataset_source is not None else None

    def _setup_params(self):
        r"""
        Setup training models and parameters to optimize.

        This method will call all `setup_XXX_params` methods in the subclass. Each `setup_XXX_params` method should return a list of training models and a list of parameters to optimize.
        """
        training_models = []
        params_to_optimize = []

        if self.use_deepspeed:
            self.logger.info("use Deepspeed")
            self.models_to_prepare = {}

        num_trainable_params = 0
        for model_params_setter in dir(self):
            if model_params_setter.startswith("setup_") and model_params_setter.endswith('_params') and callable(getattr(self, model_params_setter)):
                self.logger.info(f"setting up {model_params_setter[6:-7]} parameters...")
                try:
                    training_models_, params_to_optimize_ = getattr(self, model_params_setter)()
                except NotImplementedError:
                    continue

                for model in training_models_:
                    n_params = sum(p.numel() for p in model.parameters())
                    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    self.logger.info(
                        f"{model.__class__.__name__}: {logging.green(n_trainable_params)} / {n_params} trainable parameters")
                    num_trainable_params += n_trainable_params
                training_models.extend(training_models_)
                params_to_optimize.extend(params_to_optimize_)

        if self.mixed_precision == 'fp16' and not self.use_deepspeed:
            train_utils.patch_accelerator_for_fp16_training(self.accelerator)
        self.training_models, self.params_to_optimize = training_models, params_to_optimize
        self.num_train_params = num_trainable_params

        if self.use_ema:
            self.setup_ema()

        device_utils.clean_memory_on_device(self.accelerator.device)

    def _setup_optims(self):
        r"""
        Setup optimizer and learning rate scheduler.
        """
        # self.logger.debug(f"Number of trainable parameters: {logging.yellow(sum([p['params'].numel() for p in self.params_to_optimize]))}")

        optimizer = train_utils.get_optimizer(
            optimizer_type=self.optimizer_type,
            trainable_params=self.params_to_optimize,
            lr=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            **self.optimizer_kwargs
        )
        lr_scheduler = train_utils.get_scheduler_fix(
            lr_scheduler_type=self.lr_scheduler_type,
            optimizer=optimizer,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.lr_warmup_steps,
            num_cycles=self.lr_scheduler_num_cycles,
            power=self.lr_scheduler_power,
            **self.lr_scheduler_kwargs
        )

        self.logger.info(f"Optimizer number of parameters: {logging.yellow(sum(p.numel() for p in optimizer.param_groups[0]['params']))}")

        if self.use_deepspeed:
            from ..utils import deepspeed_utils
            ds_model = deepspeed_utils.prepare_deepspeed_model(self, **self.models_to_prepare)
            # ds_model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            #     ds_model, optimizer, lr_scheduler, self.train_dataloader
            # )
            ds_model, optimizer = deepspeed_utils.deepspeed_initialize(
                args=self,
                logger=self.logger,
                model=ds_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                dataloader=self.train_dataset,
                deepspeed_config=self.get_deepspeed_config(),
            )
            self.optimizer, self.lr_scheduler = None, lr_scheduler
            self.training_models = [ds_model]
            self.ds_model = ds_model
        else:
            self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
                optimizer, lr_scheduler, self.train_dataloader
            )

        if 'stochastic' in self.optimizer_type.lower():
            from torchastic import StochasticAccumulator
            for model in self.training_models:
                if model.dtype == torch.bfloat16:
                    StochasticAccumulator.assign_hooks(model)
                    self.logger.info(f"Stochastic accumulator assigned to {model.__class__.__name__}")

    def _setup_loss_recorder(self):
        r"""
        Setup loss recorder.
        """
        self.loss_recorder = self.get_loss_recorder()
        self.wandb_run = self.get_wandb() if self.use_wandb else None

    def get_loss_recorder(self):
        return train_utils.LossRecorder(
            gamma=self.loss_recorder_kwargs.gamma, max_window=min(self.num_steps_per_epoch, 10000)
        )

    def get_wandb(self):
        import wandb

        if self.wandb_token:
            wandb.login(key=self.wandb_token)

        if not self.wandb_project:
            self.wandb_project = self.__class__.__name__
            self.logger.info(f"No wandb project specified, auto set to {self.wandb_project}")

        if not self.wandb_name:
            self.wandb_name = os.path.basename(self.output_dir)
            self.logger.info(f"No wandb name specified, auto set to {self.wandb_name}")

        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_name,
            config=self.config,
            notes=self.note,
            resume=self.wandb_resume,
            id=self.wandb_id,
            dir=os.path.join(self.output_dir, self.output_subdir.logs),
            tags=self.wandb_tags,
            save_code=self.wandb_save_code,
        )
        run.watch(self.training_models, log='all')
        return run

    def _setup_train_state(self):
        r"""
        Setup train state.
        """
        self.train_state = self.get_train_state()
        self.train_state.setup()
        self.train_state.resume()

    def get_train_state(self):
        return self.train_state_class.from_module(self)

    def _setup_training(self):
        r"""
        Setup training information and parameter, e.g. total batch size, number of training steps, etc.
        """
        total_batch_size = self.get_total_batch_size()
        num_train_epochs = self.num_train_epochs
        num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps / self.accelerator.num_processes)
        num_train_steps = num_train_epochs * num_steps_per_epoch
        self.total_batch_size, self.num_train_epochs, self.num_steps_per_epoch, self.num_train_steps = total_batch_size, num_train_epochs, num_steps_per_epoch, num_train_steps

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"Training seed set to {logging.yellow(self.seed)}")

    def get_accelerator(self):
        # This method cannot use self.logger because the logger is not initialized yet.
        log_dir = os.path.join(self.output_dir, self.output_subdir.logs)
        log_dir = log_dir + "/" + time.strftime("%Y%m%d%H%M%S", time.localtime())
        os.makedirs(log_dir, exist_ok=True)
        assert self.mixed_precision in ['fp16', 'bf16', 'no'], \
            f"mixed_precision must be one of ['fp16', 'bf16', 'no'], but got {self.mixed_precision}, self.config.mixed_precision={self.config.mixed_precision}"
        log_with = []
        if self.use_tensorboard:
            log_with.append('tensorboard')
        if self.use_wandb:
            log_with.append('wandb')
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=log_with,
            project_dir=log_dir,
            cpu=self.cpu,
            deepspeed_plugin=deepspeed_utils.prepare_deepspeed_plugin(self)
        )
        return accelerator

    def get_logger(self):
        logger = logging.get_logger("train", disable=not self.accelerator.is_main_process)
        logging.set_all_loggers_disable(not self.accelerator.is_main_process)
        return logger

    def get_deepspeed_config(self):
        if self.zero_stage == 2:
            deepspeed_config = {
                "train_batch_size": self.get_total_batch_size(),
                "train_micro_batch_size_per_gpu": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                # "steps_per_print": args.log_every,
                "optimizer": {
                    "type": self.optimizer_type,
                    "params": {
                        "lr": self.learning_rate,
                        **self.optimizer_kwargs,
                    }
                },

                "zero_optimization": {
                    "stage": 2,
                    "reduce_scatter": False,
                    "reduce_bucket_size": 1e9,
                },

                "gradient_clipping": 1.0,
                "prescale_gradients": True,

                "fp16": {
                    "enabled": self.mixed_precision == 'fp16',
                    "loss_scale": 0,
                    "loss_scale_window": 500,
                    "hysteresis": 2,
                    "min_loss_scale": 1e-3,
                    "initial_scale_power": 15
                },

                "bf16": {
                    "enabled": False
                },

                "wall_clock_breakdown": False
            }
            if self.cpu_offloading == True:
                deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
                deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

        elif self.zero_stage == 3:
            deepspeed_config = {
                "train_batch_size": self.get_total_batch_size(),
                # "train_micro_batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                # "steps_per_print": args.log_every,

                "optimizer": {
                    "type": self.optimizer_type,
                    "params": {
                        "lr": self.learning_rate,
                        **self.optimizer_kwargs,
                    }
                },

                "zero_optimization": {
                    "stage": 3,
                    "allgather_partitions": True,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "contiguous_gradients": True,
                    "stage3_prefetch_bucket_size": 5e8,
                    "stage3_max_live_parameters": 6e8,
                    "reduce_bucket_size": 1.2e9,
                    "sub_group_size": 1e9,
                    "sub_group_buffer_num": 10,
                    "pipeline_optimizer": True,
                    "max_contigous_event_size": 0,
                    "cache_sub_group_rate": 0.0,
                    "prefetch_cache_sub_group_rate": 1.0,
                    "max_contigous_params_size": -1,
                    "max_param_reduce_events": 0,
                    "stage3_param_persistence_threshold": 9e9,
                    "is_communication_time_profiling": False,
                    "save_large_model_multi_slice": True,
                    "use_fused_op_with_grad_norm_overflow": False,
                },

                "gradient_clipping": 1.0,
                "prescale_gradients": False,

                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 500,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                    "initial_scale_power": 15
                },

                "bf16": {
                    "enabled": False
                },

                "wall_clock_breakdown": False,
                "mem_chunk": {
                    "default_chunk_size": 536870911,
                    "use_fake_dist": False,
                    "client": {
                        "mem_tracer": {
                            "use_async_mem_monitor": True,
                            "warmup_gpu_chunk_mem_ratio": 0.8,
                            "overall_gpu_mem_ratio": 0.8,
                            "overall_cpu_mem_ratio": 1.0,
                            "margin_use_ratio": 0.8,
                            "use_fake_dist": False
                        },
                        "opts": {
                            "with_mem_cache": True,
                            "with_async_move": True
                        }
                    }
                }
            }
            if self.cpu_offloading == True:
                deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
                deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

        else:
            raise ValueError

        return deepspeed_config

    def get_total_batch_size(self):
        return self.accelerator.num_processes * self.batch_size * self.gradient_accumulation_steps

    def save_config(self, fp):
        config = {k: str(self.__getattribute__(k)) for k in self.get_config().keys()}
        with open(fp, 'w') as f:
            json.dump(config, f, indent=4)
        return fp

    def save_note(self, fp):
        with open(fp, 'w') as f:
            f.write(self.note)
        return fp

    def get_dtypes(self):
        weight_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'no': torch.float32}[self.mixed_precision]
        save_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'no': torch.float32, 'float': torch.float32}[self.save_precision]
        return {'weight_dtype': weight_dtype, 'save_dtype': save_dtype}

    def get_dataset(self, setup=True, **kwargs) -> BaseDataset:
        dataset = self.dataset_class.from_config(
            self.config,
            self.accelerator,
            **kwargs
        )
        if setup:
            dataset.setup()
        return dataset

    def get_dataloader(self, dataset, shuffle=True):
        assert isinstance(dataset, BaseDataset), f"dataset must be an instance of BaseDataset, but got {type(dataset)}"
        from torch.utils.data import DataLoader

        if self.seed is not None:
            def worker_init_fn(worker_id):
                worker_seed = self.accelerator.local_process_index + self.seed
                random.seed(worker_seed)
                np.random.seed(worker_seed)
                torch.manual_seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=min(self.max_dataloader_n_workers, os.cpu_count()),
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            persistent_workers=self.persistent_data_loader_workers,
            worker_init_fn=worker_init_fn if self.seed is not None else None,
        )

    def setup_ema(self):
        raise NotImplementedError(f"setup_ema is not implemented for {self.__class__.__name__}")

    def load_models(self):
        models = {}
        model_loaders = self.get_model_loaders()
        self.logger.info(f"loading models with the following order: {' -> '.join(ml.__name__[5:-6] for ml in model_loaders)}")
        for ml in model_loaders:
            m = ml()
            models.update(m)
            for key, model in m.items():
                if key not in self.__dict__:
                    self.__dict__[key] = model
        return models

    def get_model_loaders(self):
        return [getattr(self, ml) for ml in dir(self) if ml.startswith("load_") and ml.endswith('_model') and callable(getattr(self, ml))]

    def _prepare_one_model(self, model, train, name=None, dtype=None, transform_model_if_ddp=True, device_placement=None):
        # Ensure weight dtype when full fp16/bf16 training
        if dtype is not None:
            model.to(dtype)
        elif self.full_fp16:
            assert (
                self.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16'"
            self.logger.info(f"enable full fp16 training for {model.__class__.__name__}.")
            model.to(self.weight_dtype)
        elif self.full_bf16:
            assert (
                self.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16'"
            self.logger.info(f"enable full bf16 training for {model.__class__.__name__}.")
            model.to(self.weight_dtype)
        model.to(self.device)
        if train:
            if self.use_deepspeed:
                assert name is not None, f"model name must be provided for Deepspeed training"
                assert name not in self.models_to_prepare, f"duplicate model name {name}"
                self.models_to_prepare[name] = model
            else:
                model = self.accelerator.prepare(model, device_placement=device_placement)
            if transform_model_if_ddp:
                model, = train_utils.transform_models_if_DDP([model])
        return model

    def get_loss_weight(self, batch):
        raise NotImplementedError

    def _print_start_training_message(self):
        messages = self.get_start_training_message()
        self.logger.info(messages[0], no_prefix=True)
        for message in messages[1:]:
            self.logger.info(message)

    def get_start_training_message(self):
        messages = []
        messages.append(logging.title(logging.green("Start Training")))
        messages.append(f"  num train steps: {logging.yellow(self.num_train_epochs)} x {logging.yellow(self.num_steps_per_epoch)} = {logging.yellow(self.num_train_steps)}")

        n_param_str = f"{self.num_train_params} = {self.num_train_params / 1e6:.3f}M" if self.num_train_params < 1e9 else f"{self.num_train_params} = {self.num_train_params / 1e9:.3f}B"
        messages.append(f"  number of trainable parameters: {self.num_train_params} = {n_param_str}")
        messages.append(
            f"  total batch size: {logging.yellow(self.total_batch_size)} = {self.batch_size} (batch size) x {self.gradient_accumulation_steps} (gradient accumulation steps) x {self.accelerator.num_processes} (num processes)"
        )
        messages.append(f"  mixed precision: {self.mixed_precision} | weight-dtype: {self.weight_dtype} | save-dtype: {self.save_dtype}")
        messages.append(f"  optimizer: {self.optimizer_type}")
        messages.append(f"  device: {logging.yellow(self.device)}")
        if self.use_deepspeed:
            messages.append(f"  use Deepspeed")
        return messages

    def train(self):
        self._print_start_training_message()
        self.pbar = self.train_state.pbar()
        self.pbar_logs = {}
        self.accelerator_logs = {}
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers('finetune', init_kwargs={})

        if self.train_state.save_on_train_start:
            self.train_state.save()  # on train start
        if self.train_state.eval_on_train_start:
            self.train_state.eval()  # on train start
        self.train_state.trigger_events()

        try:
            self.train_loop()
        except KeyboardInterrupt:
            save_on_train_end = self.accelerator.is_main_process and self.train_state.save_on_keyboard_interrupt
            eval_on_train_end = self.accelerator.is_main_process and self.train_state.eval_on_keyboard_interrupt
            self.logger.error("KeyboardInterrupted.")
        except Exception as e:
            import traceback
            save_on_train_end = self.accelerator.is_main_process and self.train_state.save_on_exception
            eval_on_train_end = self.accelerator.is_main_process and self.train_state.eval_on_exception
            self.logger.error("Exception:", e)
            traceback.print_exc()
        else:
            save_on_train_end = self.accelerator.is_main_process and self.train_state.save_on_train_end
            eval_on_train_end = self.accelerator.is_main_process and self.train_state.eval_on_train_end

        self.pbar.close()
        self.accelerator.wait_for_everyone()
        if save_on_train_end:
            self.logger.info(f"Saving on train end...")
            self.train_state.save(on_train_end=True)
        if eval_on_train_end:
            self.logger.info(f"Evaluating on train end...")
            self.train_state.eval(on_train_end=True)
        self.train_state.trigger_events()
        self.accelerator.end_training()
        self.logger.info(logging.green(f"Training finished at process {self.accelerator.local_process_index+1}/{self.accelerator.num_processes}"), disable=False)
        del self.accelerator

    def clip_grad_norm(self):
        if self.max_grad_norm:
            params_to_clip = []
            for m in self.training_models:
                params_to_clip.extend(m.parameters())
            self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)

    def optimizer_step(self, loss):
        # Stochastic rounding
        if 'stochastic' in self.optimizer_type.lower():
            for model in self.training_models:
                if model.dtype == torch.bfloat16:
                    from torchastic import StochasticAccumulator
                    StochasticAccumulator.reassign_grad_buffer(model)

        self.optimizer.step()

    def lr_scheduler_step(self):
        self.lr_scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        if self.use_deepspeed:
            self.ds_model.backward(loss)
            last_batch_iteration = (self.train_state.global_step + 1) // (self.total_batch_size // (self.batch_size * self.accelerator.num_processes))
            self.ds_model.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})
        else:
            self.accelerator.backward(loss)

    def train_loop(self):
        while self.train_state.epoch < self.num_train_epochs:
            self.train_dataset.shuffle()
            if self.accelerator.is_main_process:
                self.pbar.write(f"epoch: {self.train_state.epoch}/{self.num_train_epochs}")
            for m in self.training_models:
                m.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(*self.training_models) if not self.use_deepspeed else contextlib.nullcontext():
                    loss = self.train_step(batch)
                    # if self.loss_weight_getter is not None:
                    #     loss = loss * self.get_loss_weight(batch, loss)

                    self.backward(loss)

                    # verify gradient flow
                    # for m in self.training_models:
                    #     for name, param in m.named_parameters():
                    #         if param.grad is None:
                    #             self.logger.info(f"grad of {logging.blue(name)}: {logging.green(param.grad.abs().mean().item()) if param.grad is not None else logging.red(None)}")

                    self.clip_grad_norm()
                    self.optimizer_step(loss)
                    self.lr_scheduler_step()
                    self.zero_grad()

                    if self.use_ema:
                        self.ema_step()

                if self.accelerator.sync_gradients:
                    self.pbar.update(1)
                    self.train_state.step()
                    self.train_state.save(on_step_end=True)
                    self.train_state.eval(on_step_end=True)
                    self.train_state.trigger_events()

                # loggings
                step_loss: float = loss.detach().item()
                self.loss_recorder.add(loss=step_loss)
                avr_loss: float = self.loss_recorder.moving_average(window=self.loss_recorder_kwargs.stride)
                ema_loss: float = self.loss_recorder.ema
                self.accelerator_logs.update({"loss/step": step_loss, 'loss_avr/step': avr_loss, 'loss_ema/step': ema_loss})
                self.accelerator.log(self.accelerator_logs, step=self.train_state.global_step)
                if self.use_wandb:
                    self.wandb_run.log(self.accelerator_logs, step=self.train_state.global_step)
                self.pbar_logs.update({
                    'lr': self.lr_scheduler.get_last_lr()[0],
                    'epoch': self.train_state.epoch,
                    'global_step': self.train_state.global_step,
                    'next': len(self.train_dataloader) - step - 1,
                    'step_loss': step_loss,
                    'avr_loss': avr_loss,
                    'ema_loss': ema_loss,
                })
                self.pbar.set_postfix(self.pbar_logs)
                if self.gc_every_n_steps and self.train_state.global_step % self.gc_every_n_steps == 0:
                    gc.collect()

            # end of epoch
            self.accelerator_logs.update({"loss/epoch": self.loss_recorder.moving_average(window=self.num_steps_per_epoch)})
            self.accelerator.log(self.accelerator_logs, step=self.train_state.epoch)
            self.accelerator.wait_for_everyone()
            if self.use_wandb:
                self.wandb_run.log(self.accelerator_logs, step=self.train_state.epoch)
            self.train_state.save(on_epoch_end=True)
            self.train_state.eval(on_epoch_end=True)
            self.train_state.trigger_events()
            if self.gc_every_n_epochs and self.train_state.epoch % self.gc_every_n_epochs == 0:
                gc.collect()
            if self.train_state.global_step >= self.num_train_steps:
                break

    def train_step(self, batch):
        r"""
        Single training step that returns the loss.
        """
        raise NotImplementedError(f"train_step is not implemented for {self.__class__.__name__}")
