import torch
import importlib
import warnings
import transformers
from typing import Optional
from torch.optim.optimizer import Optimizer
from transformers import SchedulerType
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from . import log_utils, advanced_train_utils

logger = log_utils.get_logger("train")


VAE_SCALE_FACTOR = 0.18215


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

    # if name == SchedulerType.PIECEWISE_CONSTANT:
    #     return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

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


def get_input_ids(caption, tokenizer, max_token_length):
    input_ids = tokenizer(
        caption, padding="max_length", truncation=True, max_length=max_token_length, return_tensors="pt"
    ).input_ids

    if max_token_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        ids_list = []
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # v1
            # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
            # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
            for i in range(0, max_token_length + 2 - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                ids_list.append(ids_chunk)
        else:
            # v2 or SDXL
            # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
            for i in range(0, max_token_length + 2 - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = (
                    input_ids[0].unsqueeze(0),  # BOS
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )  # PAD or EOS
                ids_chunk = torch.cat(ids_chunk)

                # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                    ids_chunk[-1] = tokenizer.eos_token_id
                # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id

                ids_list.append(ids_chunk)

        input_ids = torch.stack(ids_list)  # 3,77
        return input_ids


def get_hidden_states(input_ids, tokenizer, text_encoder, weight_dtype=None, v2=False, clip_skip=None, max_token_length=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length is not None:
        if v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                states_list.append(encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def apply_weighted_noise(noise, mask, weight, normalize=True):
    # noise is [H, W, C] and mask is [H, W]
    mask = torch.stack([mask] * noise.shape[1], dim=1)
    noise = torch.where(mask > 0, noise * weight, noise)
    if normalize:
        noise = noise / noise.std()
    return noise


def n_params(module):
    return sum(param.numel() for param in module.parameters())


def ignore_warnings(categories=[FutureWarning, DeprecationWarning]):
    for category in categories:
        warnings.filterwarnings("ignore", category=category)


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
