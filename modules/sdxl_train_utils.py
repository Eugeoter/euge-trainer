import torch
import torch.nn as nn
import math
import os
import time
import gc
import importlib
import transformers
from accelerate import init_empty_weights, Accelerator
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from . import advanced_train_utils, sdxl_original_unet, sdxl_model_utils, model_utils, log_utils

logger = log_utils.get_logger("train")

VAE_SCALE_FACTOR = 0.13025

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def load_tokenizers(tokenizer_cache_dir=None, max_token_length=75):
    original_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokeniers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.print(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    if max_token_length is not None:
        logger.print(f"update token length: {max_token_length}")

    return tokeniers


def collate_fn(batch):
    return batch[0]


def prepare_accelerator(config):
    log_dir = os.path.join(config.output_dir, config.output_subdir.logs)
    log_dir = log_dir + "/" + time.strftime("%Y%m%d%H%M%S", time.localtime())
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with='tensorboard',
        project_dir=log_dir,
        cpu=config.cpu,
    )
    return accelerator


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


def load_target_model(config, accelerator, model_version: str, weight_dtype):
    # load models for each process
    model_dtype = match_mixed_precision(config, weight_dtype)  # prepare fp16/bf16
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}", disable=False)

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                config.pretrained_model_name_or_path,
                config.vae,
                model_version,
                weight_dtype,
                accelerator.device,
                model_dtype,
            )

            # work on low-ram device
            if config.cpu:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    name_or_path: str, vae_path: str, model_version: str, weight_dtype, device="cpu", model_dtype=None
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        logger.print(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_utils.load_models_from_sdxl_checkpoint(model_version, name_or_path, device, model_dtype)
    else:
        # Diffusers model is loaded to CPU
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.print(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None
                )
            except EnvironmentError as ex:
                if variant is not None:
                    logger.print("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            logger.print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        # convert to fp32 for cache text_encoders outputs
        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        state_dict = sdxl_model_utils.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel()  # overwrite unet
        sdxl_model_utils._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.print("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if vae_path is not None:
        vae = model_utils.load_vae(vae_path, weight_dtype)
        logger.print("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def set_diffusers_xformers_flag(model, valid):
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(valid)

        for child in module.children():
            fn_recursive_set_mem_eff(child)

    fn_recursive_set_mem_eff(model)


def replace_unet_modules(unet, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.print("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.print("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.print("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


def transform_models_if_DDP(models):
    # Transform text_encoder, unet and network from DistributedDataParallel
    return [model.module if type(model) == DDP else model for model in models if model is not None]


def get_optimizer(config, trainable_params):
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = config.optimizer_type.lower()
    optimizer_kwargs = config.optimizer_kwargs

    lr = config.learning_rate
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
            config.learning_rate = None

            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    logger.print(f"unet_lr and text_encoder_lr are ignored")
                    config.unet_lr = None
                    config.text_encoder_lr = None

            if config.lr_scheduler != "adafactor":
                logger.print(f"use adafactor_scheduler")
            config.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if config.max_grad_norm != 0.0:
                logger.print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0"
                )
            if config.lr_scheduler != "constant_with_warmup":
                logger.print(f"constant_with_warmup will be good")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.print(f"clip_threshold=1.0 will be good")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # 任意のoptimizerを使う
        optimizer_type = config.optimizer_type  # lowerでないやつ（微妙）
        logger.print(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    return optimizer


def get_scheduler_fix(config, optimizer: Optimizer, num_train_steps):
    """
    Unified API to get any scheduler from its name.
    """
    name = config.lr_scheduler
    num_warmup_steps: Optional[int] = config.lr_warmup_steps
    num_cycles = config.lr_scheduler_num_cycles
    power = config.lr_scheduler_power

    lr_scheduler_kwargs = config.lr_scheduler_kwargs

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(":")[1])
        # LOGGER.print("adafactor scheduler init lr", initial_lr)
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_train_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, power=power, **lr_scheduler_kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, **lr_scheduler_kwargs)


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2  # = alpha^2 / (1 - alpha^2)

    noise_scheduler.all_snr = all_snr.to(device)


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    pool2 = pool_workaround(text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i: i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i: i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def get_noise_noisy_latents_and_timesteps(config, noise_scheduler, latents):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if config.noise_offset:
        noise = advanced_train_utils.apply_noise_offset(latents, noise, config.noise_offset, config.adaptive_noise_scale)
    if config.multires_noise_iterations:
        noise = advanced_train_utils.pyramid_noise_like(
            noise, latents.device, config.multires_noise_iterations, config.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if config.min_timestep is None else config.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if config.max_timestep is None else config.max_timestep

    if config.timestep_sampler_type == "uniform":
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)  # timestep is in [min_timestep, max_timestep)
        timesteps = timesteps.long()
    elif config.timestep_sampler_type == "logit_normal":  # Rectified Flow from SD3 paper (partial implementation)
        from .rectified_flow import logit_normal
        timestep_sampler_kwargs = config.timestep_sampler_kwargs
        m = timestep_sampler_kwargs.get('loc', 0) or timestep_sampler_kwargs.get('mean', 0) or timestep_sampler_kwargs.get('m', 0) or timestep_sampler_kwargs.get('mu', 0)
        s = timestep_sampler_kwargs.get('scale', 1) or timestep_sampler_kwargs.get('std', 1) or timestep_sampler_kwargs.get('s', 1) or timestep_sampler_kwargs.get('sigma', 1)
        timesteps = logit_normal(mu=m, sigma=s, shape=(b_size,), device=latents.device)  # sample from logistic normal distribution
        timesteps = timesteps * (max_timestep - min_timestep) + min_timestep  # scale to [min_timestep, max_timestep)
        timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if config.ip_noise_gamma:
        noisy_latents = noise_scheduler.add_noise(latents, noise + config.ip_noise_gamma * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps


def apply_weighted_noise(noise, mask, weight, normalize=True):
    # noise is [H, W, C] and mask is [H, W]
    mask = torch.stack([mask] * noise.shape[1], dim=1)
    noise = torch.where(mask > 0, noise * weight, noise)
    if normalize:
        noise = noise / noise.std()
    return noise


def save_sdxl_model_during_train(
    config,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    save_path = os.path.join(config.output_dir, config.output_subdir.models, f"{os.path.basename(config.output_dir)}_ep{epoch}_step{global_step}.safetensors")
    sdxl_model_utils.save_stable_diffusion_checkpoint(
        save_path,
        text_encoder1,
        text_encoder2,
        unet,
        epoch,
        global_step,
        ckpt_info,
        vae,
        logit_scale,
        save_dtype,
    )
    return save_path


def save_train_state_during_train(
    config,
    optimizer,
    lr_scheduler,
    epoch: int,
    global_step: int,
):
    save_path = os.path.join(config.output_dir, config.output_subdir.train_state, f"{os.path.basename(config.output_dir)}_train-state_ep{epoch}_step{global_step}.pth")
    sdxl_model_utils.save_train_state(
        save_path,
        optimizer,
        lr_scheduler,
        epoch,
        global_step,
    )
    return save_path


def resume_train_state(config, optimizer, lr_scheduler):
    train_state_path = config.resume_from
    epoch, global_step = sdxl_model_utils.load_train_state(train_state_path, optimizer, lr_scheduler)
    logger.print(f"resume train state from: `{log_utils.yellow(train_state_path)}`")
    return epoch, global_step


def n_params(module):
    return sum(param.numel() for param in module.parameters())


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
        ema = self.ema * self.gamma + loss * (1 - self.gamma)
        ema_hat = ema / (1 - self.gamma ** self.t) if self.t < 500 else ema
        self.ema = ema_hat

    def moving_average(self, *, window: int) -> float:
        if len(self.losses) < window:
            window = len(self.losses)
        return sum(self.losses[-window:]) / window


class TrainState:
    r"""
    Class to manage training state.
    """

    def __init__(
        self,
        config,
        accelerator,
        optimizer,
        lr_scheduler,
        train_dataloader,
        unet,
        text_encoder1,
        text_encoder2,
        tokenizer1,
        tokenizer2,
        vae,
        logit_scale,
        ckpt_info,
        save_dtype,
    ):
        self.config = config
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.unet = unet
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.vae = vae
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info
        self.save_dtype = save_dtype

        self.setup()

    def setup(self):
        self.total_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.accelerator.num_processes
        self.num_train_epochs = self.config.num_train_epochs
        self.num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps / self.accelerator.num_processes)
        self.num_train_steps = self.num_train_epochs * self.num_steps_per_epoch

        self.output_model_dir = os.path.join(self.config.output_dir, self.config.output_subdir.models)
        self.output_train_state_dir = os.path.join(self.config.output_dir, self.config.output_subdir.train_state)

        output_name_default = os.path.basename(self.config.output_dir)
        self.output_name = dict(
            models=self.config.output_name.models or output_name_default,
            # samples=self.config.output_name.samples or output_name_default,
            train_state=self.config.output_name.train_state or output_name_default,
        )

        self.global_step = 0

    def step(self):
        self.global_step += 1

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch

    def _save_model(self):
        logger.print(f"saving model at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}.safetensors")
        sdxl_model_utils.save_stable_diffusion_checkpoint(
            fp=save_path,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder1=self.accelerator.unwrap_model(self.text_encoder1),
            text_encoder2=self.accelerator.unwrap_model(self.text_encoder2),
            epochs=self.epoch,
            steps=self.global_step,
            ckpt_info=self.ckpt_info,
            vae=self.vae,
            logit_scale=self.logit_scale,
            save_dtype=self.save_dtype,
        )
        logger.print(f"model saved to: `{log_utils.yellow(save_path)}`")

    def _save_train_state(self):
        logger.print(f"saving train state at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_train_state_dir, f"{self.output_name['train_state']}_train-state_ep{self.epoch}_step{self.global_step}")
        self.accelerator.save_state(save_path)
        logger.print(f"train state saved to: `{log_utils.yellow(save_path)}`")

    def save(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_save = False
        do_save |= bool(on_step_end and self.global_step and self.config.save_every_n_steps and self.global_step % self.config.save_every_n_steps == 0)
        do_save |= bool(on_epoch_end and self.epoch and self.config.save_every_n_epochs and self.epoch % self.config.save_every_n_epochs == 0)
        do_save |= bool(on_train_end)
        do_save &= bool(self.config.save_model or self.config.save_train_state)
        if do_save:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.config.save_model:
                    self._save_model()
                if self.config.save_train_state:
                    self._save_train_state()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            self.accelerator.wait_for_everyone()

    def sample(self, on_step_end=False, on_epoch_end=False, on_train_end=False):
        do_sample = False
        do_sample |= bool(on_step_end and self.global_step and self.config.sample_every_n_steps and self.global_step % self.config.sample_every_n_steps == 0)
        do_sample |= bool(on_epoch_end and self.epoch and self.config.sample_every_n_epochs and self.epoch % self.config.sample_every_n_epochs == 0)
        do_sample |= bool(on_train_end)
        if do_sample:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                from .sdxl_eval_utils import sample_during_train
                from .sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
                try:
                    sample_during_train(
                        pipe_class=SdxlStableDiffusionLongPromptWeightingPipeline,
                        accelerator=self.accelerator,
                        config=self.config,
                        epoch=self.epoch if on_epoch_end else None,
                        steps=self.global_step,
                        unet=self.unet,
                        text_encoder=[self.text_encoder1, self.text_encoder2],
                        vae=self.vae,
                        tokenizer=[self.tokenizer1, self.tokenizer2],
                        device=self.accelerator.device,
                    )

                except Exception as e:
                    import traceback
                    logger.print(log_utils.red("exception when sample images:", e))
                    traceback.print_exc()
                    pass
            self.accelerator.wait_for_everyone()

    def resume(self):
        if self.config.resume_from:
            self.accelerator.load_state(self.config.resume_from)
            self.global_step = self.accelerator.step
            logger.print(f"train state loaded from: `{log_utils.yellow(self.config.resume_from)}`")

    def pbar(self):
        from tqdm import tqdm
        return tqdm(total=self.num_train_steps, initial=self.global_step, desc='steps', disable=not self.accelerator.is_local_main_process)
