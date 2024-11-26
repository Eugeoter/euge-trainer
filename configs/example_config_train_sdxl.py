import os
from ml_collections import ConfigDict


# global variables
REFINE = False


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = '/path/to/model.safetensors'
    config.vae_model_name_or_path = None
    config.hf_cache_dir = None

    config.dataset_source = [
        dict(
            name_or_path="/path/to/images/dog/",
            read_attrs=True,  # 从同名 txt 中读取标注
        ),
        dict(
            name_or_path="/path/to/images/cats/",
            read_attrs=True,
        ),
    ]
    config.output_dir = 'projects/example_project'
    config.resume_from = None

    # config.dataset_full_cache_path = os.path.join(config.output_dir, 'dataset_full_cache.pkl')
    # config.dataset_cache_path = os.path.join(config.output_dir, 'dataset_cache.pkl')

    # Model Parameters
    config.no_half_vae = False
    config.use_deepspeed = False
    config.tokenizer_cache_dir = 'tokenizers'
    config.max_token_length = 225
    config.max_retries = None

    # Dataset Parameters
    config.resolution = 1024
    config.arb = True
    config.allow_crop = False
    config.flip_aug = True
    config.max_width = None
    config.max_height = None
    config.max_area = 1024 * 1024
    config.bucket_reso_step = 32
    config.max_aspect_ratio = 1.1
    config.predefined_buckets = [
        (1024, 1024),
        (1152, 896), (896, 1152),
        (1216, 832), (832, 1216),
        (1344, 768), (768, 1344),
        (1536, 640), (640, 1536),
        (1792, 576), (576, 1792),
        (2048, 512), (512, 2048),
    ]
    config.vae_batch_size = 6
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 4
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
    config.record_columns = ['weight']
    config.loss_recorder_kwargs = cfg(
        gamma=0.99,
        stride=1000,
    )
    config.save_precision = 'fp16'
    config.save_model = True
    config.save_train_state = False
    config.save_every_n_epochs = 0
    config.save_every_n_steps = 500
    config.save_on_train_start = False
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = False
    config.save_on_exception = False
    config.save_max_n_models = 5

    # Sample Parameters
    config.eval_size = 2
    config.eval_every_n_epochs = 1
    config.eval_every_n_steps = 500
    config.eval_on_train_start = True
    config.eval_sampler = 'euler_a'

    config.eval_params = cfg(
        prompt="1girl, solo, cowboy shot, white background, smile, looking at viewer, serafuku, pleated skirt",
        negative_prompt="abstract, bad anatomy, clumsy pose, signature",
        steps=28,
        batch_size=1,
        batch_count=1,
        scale=7.5,
        seed=42,
        width=832,
        height=1216,
        save_latents=False,
    )

    # Training Parameters
    config.num_train_epochs = 100
    config.batch_size = 2
    config.learning_rate = 5e-6
    config.learning_rate_nnet = None
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 500
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    config.mixed_precision = 'bf16'
    config.full_bf16 = False
    config.full_fp16 = False
    config.train_text_encoder = False
    config.learning_rate_te1 = 1e-6
    config.learning_rate_te2 = 1e-6
    config.gradient_checkpointing = True
    config.gradient_accumulation_steps = 4
    config.optimizer_type = 'Adafactor'
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    config.cpu = False

    # Advanced Parameters
    config.max_token_length = 225
    config.mem_eff_attn = False
    config.xformers = True
    config.diffusers_xformers = False
    config.sdpa = False
    config.clip_skip = None
    config.noise_offset = 0.0357
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0
    config.adaptive_noise_scale = None
    config.max_grad_norm = 0
    config.prediction_type = 'epsilon'
    config.zero_terminal_snr = True
    config.ip_noise_gamma = 0
    config.min_snr_gamma = 5
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.max_token_length = 225
    config.timestep_sampler_type = "uniform"

    # Cache Parameters
    config.cache_latents = False
    config.cache_latents_to_disk = True
    config.cache_only = False
    config.check_cache_validity = False
    config.keep_cached_latents_in_memory = False
    config.async_cache = True
    config.cache_latents_max_dataloader_n_workers = 64

    return config
