import torch
import importlib
import transformers
from typing import Optional
from torch.optim.optimizer import Optimizer
from transformers import SchedulerType
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from waifuset import logging

logger = logging.get_logger("train")


class LossRecorder:
    r"""
    Class to record better losses.
    """

    def __init__(self, gamma=0.9, max_window=None):
        self.losses = []
        self.gamma = gamma
        self.ema = 0
        self.t = 0
        self.max_window = max_window

    def add(self, *, loss: float) -> None:
        self.losses.append(loss)
        if self.max_window is not None and len(self.losses) > self.max_window:
            self.losses.pop(0)
        self.t += 1
        self.ema = self.ema * self.gamma + loss * (1 - self.gamma)

    def moving_average(self, *, window: int) -> float:
        if len(self.losses) < window:
            window = len(self.losses)
        return sum(self.losses[-window:]) / window


def deepspeed_config_from_config(config, global_batch_size):
    if config.use_zero_stage == 2:
        deepspeed_config = {
            "train_batch_size": global_batch_size,
            "train_micro_batch_size_per_gpu": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "steps_per_print": 100,
            "optimizer": {
                "type": config.optimizer_type,
                "params": {
                    "lr": config.learning_rate,
                    **config.optimizer_kwargs,
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
                "enabled": config.mixed_precision == "fp16",
                "loss_scale": 0,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1e-3,
                "initial_scale_power": 15
            },

            "bf16": {
                "enabled": config.mixed_precision == "bf16",
                "loss_scale": 0,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1e-3,
                "initial_scale_power": 15
            },

            "wall_clock_breakdown": False
        }
        if config.cpu_offloading == True:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

    elif config.use_zero_stage == 3:
        deepspeed_config = {
            "train_batch_size": config.global_batch_size,
            # "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "steps_per_print": 100,

            "optimizer": {
                "type": config.optimizer_type,
                "params": {
                    "lr": config.learning_rate,
                    **config.optimizer_kwargs,
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
        if config.cpu_offloading == True:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

    else:
        raise ValueError

    return deepspeed_config


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def transform_models_if_DDP(models):
    # Transform text_encoder, unet and network from DistributedDataParallel
    return [model.module if type(model) == DDP else model for model in models if model is not None]


def get_optimizer(
    optimizer_type,
    trainable_params,
    lr,
    lr_scheduler_type,
    **optimizer_kwargs,
):
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = optimizer_type.lower()
    optimizer = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
        logger.print(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

        if optimizer_type == "AdamW8bit".lower():
            logger.print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            logger.print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                logger.print(
                    f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
                )
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

        elif optimizer_type == "Lion8bit".lower():
            logger.print(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError:
                raise AttributeError(
                    "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later. / Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedAdamW8bit".lower():
            logger.print(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamW8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedLion8bit".lower():
            logger.print(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedLion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )

        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW32bit".lower():
        logger.print(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")
        try:
            optimizer_class = bnb.optim.PagedAdamW32bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW32bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamW32bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        logger.print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.print(f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        # check lr and lr_count, and LOGGER.print warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            logger.print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            logger.print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            logger.print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            # DAdaptation family
            # check dadaptation is installed
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError:
                raise ImportError("No dadaptation / dadaptation がインストールされていないようです")

            # set optimizer
            if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
                optimizer_class = experimental.DAdaptAdamPreprint
                logger.print(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                logger.print(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                logger.print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                logger.print(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                logger.print(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                logger.print(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                logger.print(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            # Prodigy
            # check Prodigy is installed
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("No Prodigy / Prodigy がインストールされていないようです")

            logger.print(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True
        if "warmup_init" not in optimizer_kwargs:
            optimizer_kwargs["warmup_init"] = False
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs["warmup_init"]:
            logger.print(f"set relative_step to True because warmup_init is True")
            optimizer_kwargs["relative_step"] = True
        logger.print(f"use Adafactor optimizer")
        logger.print(' | '.join([f"{k}={v}" for k, v in optimizer_kwargs.items()]))

        if optimizer_kwargs["relative_step"]:
            logger.print(f"relative_step is true")
            if lr != 0.0:
                logger.print(f"learning rate is used as initial_lr")

            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    logger.print(f"unet_lr and text_encoder_lr are ignored")

            if lr_scheduler_type != "adafactor":
                logger.print(f"use adafactor_scheduler")
            lr_scheduler_type = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            # if config.max_grad_norm != 0.0:
            #     logger.print(
            #         f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0"
            #     )
            if lr_scheduler_type != "constant_with_warmup":
                logger.print(f"constant_with_warmup will be good")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.print(f"clip_threshold=1.0 will be good")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "StochasticAdamW".lower():
        logger.print(f"use StochasticAdamW optimizer | {optimizer_kwargs}")
        from torchastic import AdamW as StochasticAdamW
        optimizer_class = StochasticAdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # 任意のoptimizerを使う
        logger.info(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    return optimizer


def get_scheduler_fix(
    lr_scheduler_type: str,
    optimizer: Optimizer,
    num_train_steps,
    num_warmup_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    **lr_scheduler_kwargs,
):
    """
    Unified API to get any scheduler from its name.
    """
    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{lr_scheduler_type} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    if lr_scheduler_type.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(lr_scheduler_type.split(":")[1])
        # LOGGER.print("adafactor scheduler init lr", initial_lr)
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    lr_scheduler_type = SchedulerType(lr_scheduler_type)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[lr_scheduler_type]

    if lr_scheduler_type == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    # if name == SchedulerType.PIECEWISE_CONSTANT:
    #     return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{lr_scheduler_type} requires `num_warmup_steps`, please provide that argument.")

    if lr_scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
        # logger.debug(f"optimizer: {optimizer}, num_warmup_steps: {num_warmup_steps}, lr_scheduler_kwargs: {lr_scheduler_kwargs}")
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_train_steps is None:
        raise ValueError(f"{lr_scheduler_type} requires `num_training_steps`, please provide that argument.")

    if lr_scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if lr_scheduler_type == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, power=power, **lr_scheduler_kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, **lr_scheduler_kwargs)


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def prepare_dtype(config):
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if config.save_precision == "fp16":
        save_dtype = torch.float16
    elif config.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif config.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def match_mixed_precision(config, weight_dtype):
    if config.full_fp16:
        assert (
            weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif config.full_bf16:
        assert (
            weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None


def n_params(module):
    return sum(param.numel() for param in module.parameters())


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    loss_type: str = "l2",
    huber_c: float = 0.1,
    alphas_cumprod=None,  # ew
    timesteps=None,  # ew & cmse
    loss_map=None,  # cmse
    c_step=None,  # ew
    sched_train_steps=None,  # ew
):

    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "cmse":
        from .advanced_train_utils import adaptive_clustered_mse_loss
        loss = adaptive_clustered_mse_loss(model_pred, target, timesteps, loss_map, reduction=reduction)
    elif loss_type == "ew":
        from .advanced_train_utils import exponential_weighted_loss
        loss = exponential_weighted_loss(model_pred, target, alphas_cumprod, timesteps, loss_map, reduction="none")
        mse_loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)

        schedule_start = 1
        schedule_move = -2
        interpolate_loss = schedule_start + (schedule_move * (c_step/sched_train_steps))
        if interpolate_loss < 0:
            interpolate_loss = 0
        loss = (loss * interpolate_loss) + (mse_loss * (1 - interpolate_loss))

    else:
        raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
    return loss
