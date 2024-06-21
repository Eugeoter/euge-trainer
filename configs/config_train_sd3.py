from typing import Any, List, Tuple
from ml_collections import ConfigDict


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = '/path/to/your/model.safetensors'
    config.dataset_source = [
        cfg(
            name_or_path='/path/to/your/images',
        ),
        # cfg(
        #     name_or_path='some/repo_id',
        #     split='train',
        #     column_mapping={'png': 'image', 'jpg': 'image', 'webp': 'image'}
        # )
    ]
    config.metadata_source = [
        '/path/to/your/metadata.json',
        # '/path/to/your/metadata.csv',
        # '/path/to/your/metadata.sqlite3',
        # '/path/to/your/txt_or_json_files/'
    ]
    config.output_dir = 'projects/my_project/'
    config.resume_from = None

    # Model Parameters
    config.vae_model_name_or_path = None
    config.no_half_vae = False
    config.max_token_length = 225
    config.hf_cache_dir = '/root/autodl-tmp/.cache/huggingface/'
    config.tokenizer_cache_dir = 'tokenizers'
    config.max_retries = None

    # Dataset Parameters
    config.resolution = 1024
    config.arb = True
    config.flip_aug = True
    config.bucket_reso_step = 32
    config.max_aspect_ratio = 1.1
    config.max_width = 2048
    config.max_height = 2048
    config.predefined_buckets = None
    config.vae_batch_size = 16
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 4
    config.persistent_data_loader_workers = False

    # OS Parameters
    config.output_subdir = cfg(
        models='models',
        train_state='train_state',
        samples='samples',
        logs='logs',
    )
    config.output_name = cfg(
        models=None,
        train_state=None,
    )
    config.loss_recorder_kwargs = cfg(
        gamma=0.9,
        stride=1000,
    )
    config.save_precision = 'fp16'
    config.save_model = True
    config.save_train_state = True
    config.save_every_n_epochs = 1
    config.save_every_n_steps = 500
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = False
    config.save_on_exception = False

    # Sample Parameters
    config.sample_benchmark = r'benchmarks/example_benchmark.json'
    config.sample_every_n_epochs = 1
    config.sample_every_n_steps = 500
    config.sample_at_first = True
    config.sample_sampler = 'euler_a'

    config.sample_params = cfg(
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

    # Training Parameters
    config.num_train_epochs = 100
    config.batch_size = 1
    config.learning_rate = 1e-5
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 100
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    config.full_bf16 = False
    config.full_fp16 = True
    config.include_t5 = False  # for sd3
    config.train_text_encoder = True
    config.learning_rate_te1 = 2e-6
    config.learning_rate_te2 = 2e-6
    config.learning_rate_te3 = 0  # for sd3
    config.gradient_checkpointing = False
    config.gradient_accumulation_steps = 16
    config.optimizer_type = 'AdaFactor'
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    config.mixed_precision = 'fp16'
    config.cpu = False

    # Advanced Parameters
    config.mem_eff_attn = False
    config.xformers = True
    config.diffusers_xformers = False
    config.sdpa = False
    config.clip_skip = None
    config.noise_offset = 0
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0.25
    config.adaptive_noise_scale = None
    config.max_grad_norm = 0.5
    config.prediction_type = 'epsilon'
    config.zero_terminal_snr = False
    config.ip_noise_gamma = 0
    config.min_snr_gamma = 5
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.timestep_sampler_type = "logit_normal"
    config.timestep_sampler_kwargs = cfg(
        mu=1.09861228867,  # log(3) for 1024x1024
        sigma=1,
    )

    # Custom Training Parameters
    config.data_preprocessor = get_img_md
    config.dataset_info_getter = get_dataset_info
    config.data_weight_getter = get_data_weight
    config.caption_getter = get_caption

    # Cache Parameters
    config.cache_latents = False
    config.cache_latents_to_disk = True
    config.check_cache_validity = False
    config.keep_cached_latents_in_memory = False
    config.async_cache = True

    return config


def get_img_md(img_md, dataset_info, **kwargs):
    r"""
    Pre-process the metadata of the image.
    """
    return img_md


def get_dataset_info(dataset, **kwargs) -> Any:
    r"""
    Get the information of the dataset.
    You can iterate over `dataset.data.items()` to get the metadata of each data (row of your metadata file).
    """
    return None


def get_data_weight(img_md, dataset_info, **kwargs) -> int:
    r"""
    Get the number of repeats of a data whin a single epoch. If returns 0, the data is skipped.
    """
    return 1


def get_caption(
    img_md,
    dataset_info,
    **kwargs,
) -> str:
    r"""
    Post-process a caption when the data is loaded by the train dataloader.
    """
    return img_md['caption']
