import os
import torch
import argparse
from functools import partial
from accelerate import DeepSpeedPlugin, Accelerator
from waifuset import logging

logger = logging.get_logger("deepspeed")


def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.use_deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)
    if args.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0  # preventing overflow.
    if args.full_fp16 or args.fp16_master_weights_and_gradients:
        if args.offload_optimizer_device == "cpu" and args.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin


# Accelerate library does not support multiple models for deepspeed. So, we need to wrap multiple models into a single model.
def prepare_deepspeed_model(args: argparse.Namespace, **models):
    # remove None from models
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()
            self.models = torch.nn.ModuleDict()

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)
                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"
                self.models.update(torch.nn.ModuleDict({key: model}))

        def get_models(self):
            return self.models

    ds_model = DeepSpeedWrapper(**models)
    return ds_model


def deepspeed_config_from_args(args, global_batch_size):
    if args.use_zero_stage == 2:
        deepspeed_config = {
            "train_batch_size": global_batch_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accu_steps,
            "steps_per_print": args.log_every,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-08,
                    "weight_decay": args.weight_decay
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
                "enabled": args.use_fp16,
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
        if args.cpu_offloading == True:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

    elif args.use_zero_stage == 3:
        deepspeed_config = {
            "train_batch_size": args.global_batch_size,
            # "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accu_steps,
            "steps_per_print": args.log_every,

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-08,
                    "weight_decay": args.weight_decay
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
        if args.cpu_offloading == True:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            deepspeed_config["zero_optimization"]["offload_parameter"] = {"device": "cpu", "pin_memory": True}

    else:
        raise ValueError

    return deepspeed_config


def get_trainable_params(model):
    params = model.parameters()
    params = [p for p in params if p.requires_grad]
    return params


def deepspeed_initialize(args, logger, model, optimizer, lr_scheduler, dataloader, deepspeed_config):
    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    logger.info(f"Initialize deepspeed...")
    logger.info(f"    Using deepspeed optimizer")

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=get_trainable_params(model),
        config_params=deepspeed_config,
        args=args,
        lr_scheduler=lr_scheduler,
        training_data=dataloader
    )
    return model, optimizer
