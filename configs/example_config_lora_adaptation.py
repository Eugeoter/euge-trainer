from ml_collections import ConfigDict
from tools.control import *
from tools.dataset.coco import get_coco2017_caption


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = 'botp/stable-diffusion-v1-5'
    # config.vae_model_name_or_path = '/root/autodl-tmp/models/sdxl_vae.safetensors'
    config.pretrained_lora_model_name_or_path = {
        'path': '/root/autodl-tmp/models/sd15_lora/ghibli_style.safetensors',
        'tau_phi': 'ghibli style',
        'tau_psi': 'ghibli style',
    }

    config.backbone_type = 'sd15'
    config.init_w0 = 1.0
    config.init_w1 = 1.0
    config.lora_strength = 1.0
    config.lambda_lpips = 0
    config.lr_w0 = 1e-4
    config.lr_w1 = 5e-3
    config.loss_beta_1 = 0.5
    config.loss_beta_2 = 0.5

    config.use_wandb = True

    config.seed = 114514

    config.dataset_source = [
        dict(
            name_or_path='/root/autodl-tmp/datasets/COCO2017',
            split='train2017',
            dataset_type='coco',
        ),
    ]
    config.valid_dataset_source = [
        dict(
            name_or_path='/root/autodl-tmp/datasets/COCO2017',
            split='val2017',
            dataset_type='coco',
        ),
    ]
    config.caption_getter = get_coco2017_caption

    config.output_dir = 'projects/lora_adaptation/ghibli-layerwise'
    config.resolution = 512
    config.allow_crop = False
    config.random_crop = False
    config.arb = True
    config.flip_aug = True

    # Model Parameters
    config.no_half_vae = False
    config.use_deepspeed = False
    config.tokenizer_cache_dir = 'tokenizers'
    config.max_token_length = 225
    config.hf_cache_dir = None
    config.max_retries = None

    # Dataset Parameters
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
    config.save_train_state = False
    config.save_every_n_epochs = 0
    config.save_every_n_steps = 500
    config.save_on_train_start = False
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = False
    config.save_on_exception = False
    config.save_max_n_models = 2
    config.save_max_n_train_states = 1

    # Sample Parameters
    # config.eval_benchmark = r'benchmarks/example_benchmark.json'
    config.eval_train_size = 4
    config.eval_valid_size = 4
    config.eval_every_n_epochs = 0
    config.eval_every_n_steps = 100
    config.eval_on_train_start = False
    config.eval_sampler = 'euler_a'

    config.eval_params = cfg(
        prompt="1girl, solo, cowboy shot, white background, smile, looking at viewer, serafuku, pleated skirt",
        negative_prompt="abstract, bad anatomy, clumsy pose, signature",
        steps=28,
        batch_size=1,
        batch_count=1,
        scale=7.5,
        seed=42,
        width=512,
        height=512,
        save_latents=False,
        control_scale=1,
    )

    # Training Parameters
    config.num_train_epochs = 1
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.train_nnet = False
    config.learning_rate_nnet = 0
    config.lr_scheduler = 'constant_with_warmup'
    # config.lr_scheduler = 'cosine_with_restarts'
    config.lr_warmup_steps = 50
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 5
    config.lr_scheduler_kwargs = cfg()
    config.mixed_precision = 'fp16'
    config.full_bf16 = False
    config.full_fp16 = True
    config.train_text_encoder = False
    config.learning_rate_te = 0
    config.gradient_checkpointing = False
    config.gradient_accumulation_steps = 1
    config.optimizer_type = 'Adafactor'
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        # weight_decay=0.03,
        # betas=(0.9, 0.9),
        # amsgrad=False
    )
    config.cpu = False

    # Advanced Parameters
    config.max_token_length = 225
    config.mem_eff_attn = False
    config.xformers = True
    config.diffusers_xformers = False
    config.sdpa = False
    config.clip_skip = None
    config.noise_offset = 0
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0
    config.adaptive_noise_scale = None
    config.max_grad_norm = None
    config.prediction_type = 'epsilon'
    config.zero_terminal_snr = False
    config.ip_noise_gamma = 0
    config.min_snr_gamma = 0
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.max_token_length = 225
    config.timestep_sampler_type = 'uniform'

    # Cache Parameters
    config.cache_latents = False
    config.cache_latents_to_disk = True
    config.cache_only = False
    config.check_cache_validity = False
    config.keep_cached_latents_in_memory = False
    config.async_cache = True
    config.cache_latents_max_dataloader_n_workers = 64

    return config
