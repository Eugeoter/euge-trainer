import os
from tools.waifu import eugebooru
from ml_collections import ConfigDict


# global variables
REFINE = False


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = '/root/autodl-tmp/models/noob_checkpoint-e3_s42000.safetensors'
    config.vae_model_name_or_path = '/root/autodl-tmp/models/sdxl_vae_fp16_fix.safetensors'
    config.hf_cache_dir = '/root/autodl-tmp/cache/huggingface/'

    # config.dataset_source = [
    #     '/root/autodl-tmp/datasets/eugebooru/aid_all-md-2024-10-12-1.sqlite3',
    #     '/root/autodl-tmp/datasets/eugebooru/train',  # images
    #     '/root/autodl-tmp/datasets/eugebooru/cogvlm.sqlite3',
    # ]
    config.dataset_source = [
        '/root/autodl-tmp/datasets/eugebooru/metadata-0-2024-10-12-1.sqlite3',
        '/root/autodl-tmp/datasets/eugebooru/train/preparation-0',  # images
        '/root/autodl-tmp/datasets/eugebooru/cogvlm.sqlite3',
    ]
    # config.dataset_source = [
    #     '/root/autodl-tmp/datasets/multicepts/metadata.sqlite3',
    #     '/root/autodl-tmp/datasets/multicepts/images/by yomu',  # images
    #     '/root/autodl-tmp/datasets/multicepts/images/by dino',
    # ]
    config.output_dir = 'projects/awa21-aa'
    # config.resume_from = '/root/autodl-tmp/sd-trainer-0613/projects/sdxl-awa2-al/train_state/sdxl-awa2-al_train-state_ep0_step1000'

    config.dataset_full_cache_path = os.path.join(config.output_dir, 'dataset_full_cache.pkl')
    config.dataset_cache_path = os.path.join(config.output_dir, 'dataset_cache.pkl')

    # Custom Training Parameters
    config.data_preprocessor = eugebooru.get_img_md
    config.dataset_hook_getter = eugebooru.get_dataset_hook
    config.dataset_hook_saver = eugebooru.save_dataset_hook
    config.data_weight_getter = eugebooru.get_data_weight
    config.caption_getter = eugebooru.get_caption
    config.loss_weight_getter = eugebooru.get_loss_weight

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
    # config.eval_benchmark = r'benchmarks/example_benchmark.json'
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
    config.learning_rate = 2e-5
    config.train_nnet = True
    config.learning_rate_nnet = None
    # config.block_lrs = [
    #     5e-6,  # time embed & label embed
    #     5e-6, 3.75e-6, 2.5e-6,  # in_blocks.0
    #     3.75e-6, 2.5e-6, 2e-6,  # in_blocks.1
    #     2e-6, 2e-6, 2e-6,  # in_blocks.2
    #     5e-6, 5e-6, 5e-6,  # mid_block
    #     7.5e-6, 7.5e-6, 7.5e-6,  # out_blocks.0
    #     8.5e-6, 9.25e-6, 1e-5,  # out_blocks.1
    #     2.5e-6, 2.5e-6, 2.5e-6,  # out_blocks.2
    #     2.5e-6,  # out
    # ]
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 500
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    # config.nnet_trainable_params = [r'.*output_blocks.*', r'.*\.out\..*']
    config.full_bf16 = False
    config.full_fp16 = True
    config.train_text_encoder = True
    config.learning_rate_te1 = 1e-5
    config.learning_rate_te2 = 1e-5
    config.gradient_checkpointing = True
    config.gradient_accumulation_steps = 4
    config.optimizer_type = 'Adafactor'
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        # weight_decay=0.03,
        # betas=(0.9, 0.9),
        # amsgrad=False
    )
    config.mixed_precision = 'fp16'
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
    config.max_grad_norm = 0
    config.prediction_type = 'v_prediction'
    config.zero_terminal_snr = True
    config.ip_noise_gamma = 0
    config.min_snr_gamma = 0
    config.debiased_estimation_loss = True
    config.min_timestep = 0
    config.max_timestep = 1000
    config.max_token_length = 225
    config.timestep_sampler_type = "uniform"
    # config.timestep_sampler_type = "logit_normal"
    # config.timestep_sampler_kwargs = cfg(
    #     mu=1.09861228867,  # log(3)
    #     sigma=1,
    # )

    # Cache Parameters
    config.cache_latents = False
    config.cache_latents_to_disk = True
    config.cache_only = False
    config.check_cache_validity = False
    config.keep_cached_latents_in_memory = False
    config.async_cache = True
    config.cache_latents_max_dataloader_n_workers = 64

    return config
