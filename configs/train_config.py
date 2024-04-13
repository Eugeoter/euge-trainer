from ml_collections import ConfigDict


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = r'/path/to/your/model.safetensors'
    config.image_dirs = [r'/path/to/your/images']
    config.metadata_files = []
    config.output_dir = 'train-1'

    # Model Parameters
    config.vae = None

    config.no_half_vae = False
    config.tokenizer_cache_dir = 'tokenizers'
    config.records_cache_dir = 'records'

    # Dataset Parameters
    config.flip_aug = True
    config.bucket_reso_step = 32
    config.resolution = 1024
    config.vae_batch_size = 1
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 4
    config.persistent_data_loader_workers = False

    # OS Parameters
    config.output_subdir = cfg(
        models='models',
        samples='samples',
        logs='logs',
    )
    config.loss_recorder_kwargs = cfg(
        gamma=0.9,
        stride=1000,
    )
    config.save_precision = 'fp16'
    config.save_every_n_epochs = None
    config.save_every_n_steps = None
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = True
    config.save_on_exception = True

    # Training Parameters
    config.num_train_epochs = 100
    config.batch_size = 1
    config.learning_rate = 1e-5
    config.block_lr = None
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 100
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    config.full_bf16 = True
    config.full_fp16 = True
    config.train_text_encoder = False
    config.learning_rate_te1 = None
    config.learning_rate_te2 = None
    config.gradient_checkpointing = False
    config.gradient_accumulation_steps = 1
    config.optimizer_type = 'AdaFactor'
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    config.mixed_precision = 'fp16'
    config.cpu = False

    # Advanced Parameters
    config.max_token_length = 225
    config.mem_eff_attn = False
    config.xformers = True
    config.diffusers_xformers = False
    config.sdpa = False
    config.clip_skip = 1
    config.noise_offset = 0.0
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0.25
    config.adaptive_noise_scale = None
    config.max_grad_norm = 0.0
    config.zero_terminal_snr = False
    config.ip_noise_gamma = 0.0
    config.min_snr_gamma = 5.0
    config.scale_v_pred_loss_like_noise_pred = False
    config.v_pred_like_loss = 0.0
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.timestep_sampler_type = "uniform"
    config.timestep_sampler_kwargs = cfg()

    # Sample Parameters
    config.sample_benchmark = r'benchmarks/example_benchmark.json'
    config.sample_every_n_epochs = None
    config.sample_every_n_steps = None
    config.sample_sampler = 'euler_a'

    config.sample_params = cfg(
        prompt="1girl, solo, cowboy shot, white background, smile, looking at viewer, serafuku, pleated skirt",
        negative_prompt="",
        steps=28,
        batch_size=1,
        batch_count=1,
        scale=7.5,
        seed=42,
        width=832,
        height=1216,
        original_width=None,
        original_height=None,
        original_scale_factor=1.0,
        save_latents=False,
    )

    # Custom Training Parameters
    config.num_repeats_getter = None
    config.caption_processor = None
    config.description_processor = None

    # Cache Parameters
    config.cache_latents = True
    config.cache_latents_to_disk = True
    config.check_cache_validity = True
    config.keep_cached_latents_in_memory = True
    config.async_cache = True

    return config
