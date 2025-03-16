import random
from ml_collections import ConfigDict
from waifuset import FastDataset, logging


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.vae_model_name_or_path = r"d:\AI\models\sdxl\vae\sdxl_vae.safetensors"
    config.dataset_source = [r"d:\AI\datasets\aid\images\preparation-0"]
    config.valid_dataset_source = r"D:\AI\datasets\aid\images\preparation-6\redum4"
    config.output_dir = r'D:\AI\projects\sd-trainer\vae\2025-01-26-0'

    config.seed = 114514

    # Model Parameters
    config.use_deepspeed = False
    # config.hf_cache_dir = '/root/autodl-tmp/.cache/huggingface/'
    config.max_retries = None

    config.use_wandb = True
    config.wandb_token = '678e544e1b7d9702bc0d5ee6e22c8c2ad33915d2'

    # Dataset Parameters
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 0
    config.persistent_data_loader_workers = False

    # OS Parameters
    config.output_subdir = cfg(
        models='models',
        train_state='train_state',
        samples='samples',
        logs='logs',
        records='records',
    )
    config.output_name = cfg(
        models=None,
        train_state=None,
    )
    config.record_columns = None
    config.loss_recorder_kwargs = cfg(
        gamma=0.9,
        stride=1000,
    )

    config.save_precision = 'float'
    config.save_model = True
    config.save_best_model = True
    config.save_train_state = False
    config.save_every_n_epochs = 0
    config.save_every_n_steps = 0
    config.save_on_train_start = False
    config.save_on_train_end = False
    config.save_on_keyboard_interrupt = False
    config.save_on_exception = False
    config.save_max_n_models = 1

    config.eval_on_train_start = True
    config.eval_every_n_steps = 1000
    config.eval_every_n_epochs = 1

    # Training Parameters
    config.num_train_epochs = 100
    config.batch_size = 1
    config.learning_rate = 5e-5
    # config.lr_scheduler = 'cosine_with_restarts'
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 500
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = config.num_train_epochs // 10
    config.lr_scheduler_kwargs = cfg()
    config.mixed_precision = 'no'
    config.full_bf16 = False
    config.full_fp16 = False
    config.gradient_checkpointing = True
    config.gradient_accumulation_steps = 1
    config.optimizer_type = 'AdamW'
    config.optimizer_kwargs = cfg(
        # relative_step=False,
        # scale_parameter=False,
        # warmup_init=False,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        # amsgrad=False
    )
    config.cpu = False

    config.lambda_lpips = 0.0
    config.lambda_kld = 1
    config.lr_discriminator = 1e-4
    config.disc_optimizer_type = 'Adafactor'
    config.disc_optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        # weight_decay=1e-2,
        # betas=(0.9, 0.95),
    )

    return config


def split_dataset(img_md, **kwargs):
    if random.random() < 0.1:
        return 'validation'
    else:
        return 'train'


QUALITY2SCORE = {
    'amazing': 1.0,
    'best': 0.85,
    'high': 0.65,
    'normal': 0.5,
    'low': 0.35,
    'worst': 0.15,
    'horrible': 0.0,
}


def get_score(img_md, **kwargs):
    quality = img_md['quality']
    return QUALITY2SCORE[quality]


def get_dataset_info(dataset, **kwargs):
    quality2count = {}
    quality2weight = {}
    for img_key, img_md in dataset.dataset.items():
        quality = img_md['quality']
        if quality not in quality2count:
            quality2count[quality] = 0
        quality2count[quality] += 1
        if quality not in quality2weight:
            quality2weight[quality] = 0
        quality2weight[quality] += img_md.get('weight', 1)
    avr_count = sum(quality2count.values()) / len(quality2count)
    max_count = max(quality2count.values())
    avr_weight = sum(quality2weight.values()) / len(quality2weight)
    max_weight = max(quality2weight.values())
    return {
        'quality2count': quality2count,
        'avr_count': avr_count,
        'max_count': max_count,
        'quality2weight': quality2weight,
        'avr_weight': avr_weight,
        'max_weight': max_weight,
    }


def get_data_weight(img_md, dataset_info, **kwargs):
    quality = img_md['quality']
    count = dataset_info['quality2count'][quality]
    benchmark = dataset_info['max_count']
    weight = benchmark / count
    weight, prob = int(weight), weight - int(weight)
    if random.random() < prob:
        weight += 1
    weight = min(max(1, weight), 50)
    return weight
