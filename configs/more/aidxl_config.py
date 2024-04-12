import random
from ml_collections import ConfigDict


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()

    # Main parameters
    config.pretrained_model_name_or_path = r'/path/to/your/model.safetensors'
    config.image_dirs = [r'/path/to/your/images']
    config.metadata_files = []
    config.output_dir = 'train-%index%'

    # Model Parameters
    config.vae = None

    config.no_half_vae = False
    config.tokenizer_cache_dir = 'tokenizers'
    config.records_cache_dir = 'records'

    # Dataset Parameters
    config.flip_aug = True
    config.bucket_reso_step = 32
    config.resolution = 1024
    config.vae_batch_size = 6
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 4
    config.persistent_data_loader_workers = False

    # OS Parameters
    config.loss_recorder_kwargs = cfg(
        gamma=0.9,
        stride=1000,
    )
    config.save_precision = 'fp16'
    config.save_every_n_epochs = 1
    config.save_every_n_steps = 1000
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = True
    config.save_on_exception = True

    # Training Parameters
    config.num_train_epochs = 100
    config.batch_size = 1
    config.learning_rate = 1e-5
    config.block_lr = None
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 250
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    config.full_bf16 = True
    config.full_fp16 = True
    config.train_text_encoder = True
    config.learning_rate_te1 = 1e-5
    config.learning_rate_te2 = 1e-5
    config.gradient_checkpointing = True
    config.gradient_accumulation_steps = 96
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
    config.clip_skip = 2
    config.noise_offset = 0.0
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0.25
    config.adaptive_noise_scale = None
    config.max_grad_norm = 0.0
    config.zero_terminal_snr = True
    config.ip_noise_gamma = 0.0
    config.min_snr_gamma = 5.0
    config.scale_v_pred_loss_like_noise_pred = False
    config.v_pred_like_loss = 0.0
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.timestep_sampler_type = "logit_normal"
    config.timestep_sampler_kwargs = cfg(
        mu=0,
        sigma=1,
    )

    # Sample Parameters
    config.sample_benchmark = r'benchmarks/example_benchmark.json'
    config.sample_every_n_epochs = 1
    config.sample_every_n_steps = 1000
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
    config.num_repeats_getter = get_num_repeats
    config.caption_processor = process_caption
    config.description_processor = None

    # Cache Parameters
    config.cache_latents = True
    config.cache_latents_to_disk = True
    config.check_cache_validity = True
    config.keep_cached_latents_in_memory = True
    config.async_cache = True

    return config


def get_num_repeats(img_key, img_md, **kwargs):
    r"""
    提高弱势类别数据的训练权重，以及图片高质量图像的权重。
    """
    artist_benchmark = 100
    character_benchmark = 1000
    min_num_repeats = 1
    max_num_repeats = 10

    counter = kwargs.get('counter')  # 获取计数器

    num_repeats = 1

    artist = fmt2danbooru(img_md['artist'])
    if artist is not None:
        cnt = counter['artist'][artist]
        if cnt >= 25:
            num_repeats *= max(1, artist_benchmark / cnt)

    characters = fmt2danbooru(img_md['characters'])
    if characters is not None:
        for character in characters:
            cnt = counter['character'][character]
            if cnt >= 50:
                num_repeats *= max(1, character_benchmark / cnt)

    quality = img_md.get('quality')
    if quality in ('horrible', 'worst', 'low'):
        return 0  # 不训练
    elif quality in QUALITY2NRP:
        num_repeats *= QUALITY2NRP[quality]

    num_repeats = int(num_repeats)
    num_repeats = min(max_num_repeats, num_repeats)
    num_repeats = max(min_num_repeats, num_repeats)
    return num_repeats


SAFE2TAG = {
    'g': 'rating: general, safe',
    's': 'rating: sensitive',
    'q': 'rating: questionable',
    'e': 'rating: explicit, nsfw',
}


def process_caption(img_info, **kwargs):
    r"""
    按以下方式处理标签：
    - 如果图像水平翻转，将标签中的“左”和“右”互换
    - 按照图像的原图尺寸，添加缩略图和低分辨率标签
    - 按照图像的安全评级，添加安全评级标签
    - 如果艺术家标签出现频率过高，则以一定概率移除之
    - 解析并还原艺术家、角色和风格标签
    """
    artist_benchmark = 100
    character_benchmark = 1000
    caption = img_info.caption
    tags = caption.split(', ')

    if (flip_aug := kwargs.get('flip_aug', False)):
        tags = [tag.replace('left', 'right').replace('right', 'left') for tag in tags]

    if (original_size := img_info.original_size):
        original_area = original_size[0] * original_size[1]
        if original_area <= 256*256:
            tags.append('thumbnail')
        elif original_area <= 640*640:
            tags.append('lowres')

    if (safe_level := img_info.safe_level) and safe_level in SAFE2TAG:
        tags.extend(SAFE2TAG[safe_level].split(', '))

    counter = kwargs['counter']
    if (artist := img_info.metadata.get('artist')) and (cnt := counter[fmt2danbooru(artist)]) > artist_benchmark:
        artist_tag = f'artist: {artist}'
        if artist_tag in tags and random.random() > artist_benchmark / cnt:
            tags.remove(artist_tag)

    tags = [tag.split(':')[1].strip() for tag in tags if tag.startswith(('character:', 'style:'))]
    tags = ['by ' + tag.split(':')[1].strip() for tag in tags if tag.startswith('artist:')]

    return ', '.join(tags)


def fmt2dan(tag):
    if isinstance(tag, str):
        tag = tag.lower().strip()
        tag = tag.replace(' ', '_').replace('\\(', '(').replace('\\)', ')').replace(': ', ':')
        return tag
    elif isinstance(tag, list):
        return [fmt2dan(t) for t in tag]
    else:
        return tag


QUALITY2NRP = {
    'amazing': 5,
    'best': 2.5,
    'high': 1.5,
    'normal': 1,
}
