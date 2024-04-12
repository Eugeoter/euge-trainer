from ml_collections import ConfigDict


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Model Parameters
    config.pretrained_model_name_or_path = None  # [必填]
    config.vae = None
    config.no_half_vae = False
    config.tokenizer_cache_dir = None
    config.max_token_length = 225
    config.mem_eff_attn = False
    config.xformers = False
    config.diffusers_xformers = False
    config.sdpa = False
    config.clip_skip = 1

    # OS Parameters
    config.output_dir = 'eval-1'
    config.full_bf16 = False
    config.full_fp16 = False
    config.mixed_precision = 'fp16'
    config.cpu = False

    # Sample Parameters
    config.sample_benchmark = './benchmarks/example_benchmark.json'
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

    return config
