import gc
import torch
import os
from safetensors.torch import load_file
from waifuset import logging
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from modules.utils import eval_utils

logger = logging.get_logger('pipeline_tools')

UNET_CONFIG = {
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 768,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ]
}


def get_scheduler(
    scheduler_name,
    scheduler_config,
):
    if scheduler_name == 'Euler a':
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_name == 'UniPC':
        from diffusers.schedulers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
    elif scheduler_name == 'Euler':
        from diffusers.schedulers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_name == 'DDIM':
        from diffusers.schedulers import DDIMScheduler
        scheduler = DDIMScheduler.from_config(scheduler_config)
    elif scheduler_name == 'DDPM':
        from diffusers.schedulers import DDPMScheduler
        scheduler = DDPMScheduler.from_config(scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return scheduler


def get_StableDiffusionPipeline(
    pretrained_model_name_or_path,
    vae_model_name_or_path=None,
    scheduler_name='Euler A',
    enable_xformers_memory_efficient_attention=False,
    hf_cache_dir=None,
    tokenizer_cache_dir=None,
    device=None,
):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from modules.utils import sd15_model_utils

    pipeline_init_kwargs = {}

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        models = sd15_model_utils.load_models_from_stable_diffusion_checkpoint(pretrained_model_name_or_path, device=device, strict=False, nnet_class=UNet2DConditionModel)
    else:
        models = sd15_model_utils.load_models_from_stable_diffusion_diffusers_state(
            pretrained_model_name_or_path,
            device=device,
            cache_dir=hf_cache_dir,
            nnet_class=UNet2DConditionModel,
        )

    tokenizer = sd15_model_utils.load_sd15_tokenizer(
        sd15_model_utils.TOKENIZER_PATH,
        subfolder=None,
        cache_dir=tokenizer_cache_dir,
        max_token_length=225,
    )
    pipeline_init_kwargs['unet'] = models['nnet']
    pipeline_init_kwargs['text_encoder'] = models['text_encoder']
    pipeline_init_kwargs.setdefault('vae', models['vae'])
    pipeline_init_kwargs['tokenizer'] = tokenizer
    pipeline: StableDiffusionPipeline = StableDiffusionPipeline(
        **pipeline_init_kwargs,
        safety_checker=None,
        feature_extractor=None,
        scheduler=eval_utils.get_sampler(scheduler_name),
        requires_safety_checker=False,
    )

    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline


def get_StableDiffusionControlNetPipeline(
    pretrained_model_name_or_path,
    controlnet_model_name_or_path,
    vae_model_name_or_path=None,
    scheduler_name='UniPC',
    enable_xformers_memory_efficient_attention=False,
    revision=None,
    variant=None,
    hf_cache_dir=None,
    use_safetensors=True,
    device=None,
):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    pipeline_init_kwargs = {}

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

    if controlnet_model_name_or_path is not None:
        pipeline_init_kwargs["controlnet"] = ControlNetModel.from_pretrained(controlnet_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        pipeline: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_single_file(
            pretrained_model_name_or_path,
            use_safetensors=pretrained_model_name_or_path.endswith(".safetensors"),
            local_files_only=True,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )
    else:
        pipeline: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            use_safetensors=use_safetensors,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )

    pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline


def get_StableDiffusionControlNeXtPipeline(
    pretrained_model_name_or_path,
    unet_model_name_or_path,
    controlnet_model_name_or_path,
    vae_model_name_or_path=None,
    scheduler_name='UniPC',
    lora_path=None,
    load_weight_increasement=False,
    enable_xformers_memory_efficient_attention=False,
    revision=None,
    variant=None,
    hf_cache_dir=None,
    use_safetensors=True,
    device=None,
):
    from modules.models.sd15.controlnext_nnet import ControlNeXtUNet2DConditionModel
    from modules.models.sd15.controlnext import ControlNeXtModel
    from modules.pipelines.controlnext_pipeline import StableDiffusionControlNeXtPipeline

    pipeline_init_kwargs = {}

    print(f"loading unet from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        # load unet from local checkpoint
        unet_sd = load_file(pretrained_model_name_or_path) if pretrained_model_name_or_path.endswith(".safetensors") else torch.load(pretrained_model_name_or_path)
        unet = ControlNeXtUNet2DConditionModel.from_config(UNET_CONFIG)
        unet.load_state_dict(unet_sd, strict=False)
    else:
        unet = ControlNeXtUNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=hf_cache_dir,
            variant=variant,
            torch_dtype=torch.float16,
            use_safetensors=use_safetensors,
            subfolder="unet",
        )
    unet = unet.to(dtype=torch.float16)
    pipeline_init_kwargs["unet"] = unet

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

    if controlnet_model_name_or_path is not None:
        pipeline_init_kwargs["controlnet"] = ControlNeXtModel().to(device, dtype=torch.float32)  # init

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        pipeline: StableDiffusionControlNeXtPipeline = StableDiffusionControlNeXtPipeline.from_single_file(
            pretrained_model_name_or_path,
            use_safetensors=pretrained_model_name_or_path.endswith(".safetensors"),
            local_files_only=True,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )
    else:
        pipeline: StableDiffusionControlNeXtPipeline = StableDiffusionControlNeXtPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            use_safetensors=use_safetensors,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )

    pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
    if unet_model_name_or_path is not None:
        print(f"loading controlnext unet from {unet_model_name_or_path}")
        pipeline.load_controlnext_unet_weights(
            unet_model_name_or_path,
            load_weight_increasement=load_weight_increasement,
            use_safetensors=True,
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )
    if controlnet_model_name_or_path is not None:
        print(f"loading controlnext controlnet from {controlnet_model_name_or_path}")
        pipeline.load_controlnext_controlnet_weights(
            controlnet_model_name_or_path,
            use_safetensors=True,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        )
    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if lora_path is not None:
        pipeline.load_lora_weights(lora_path)
    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline


def get_StableDiffusionAdapterPipeline(
    pretrained_model_name_or_path,
    t2i_adapter_model_name_or_path,
    vae_model_name_or_path=None,
    scheduler_name='UniPC',
    enable_xformers_memory_efficient_attention=False,
    revision=None,
    variant=None,
    hf_cache_dir=None,
    use_safetensors=True,
    device=None,
):
    from diffusers import T2IAdapter, StableDiffusionAdapterPipeline

    pipeline_init_kwargs = {}

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

    if t2i_adapter_model_name_or_path is not None:
        adapter = T2IAdapter.from_pretrained(
            t2i_adapter_model_name_or_path, torch_dtype=torch.float16, varient="fp16"
        ).to("cuda")
        pipeline_init_kwargs["adapter"] = adapter

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        pipeline: StableDiffusionAdapterPipeline = StableDiffusionAdapterPipeline.from_single_file(
            pretrained_model_name_or_path,
            use_safetensors=pretrained_model_name_or_path.endswith(".safetensors"),
            local_files_only=True,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )
    else:
        pipeline: StableDiffusionAdapterPipeline = StableDiffusionAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            use_safetensors=use_safetensors,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )

    pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline


def get_StableDiffusionXLPipeline(
    pretrained_model_name_or_path,
    scheduler_name='UniPC',
    enable_xformers_memory_efficient_attention=False,
    revision=None,
    variant=None,
    hf_cache_dir=None,
    device=None,
    no_half_vae=False,
    prediction_type=None,
    rescale_betas_zero_snr=None,
):
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

    pipeline_init_kwargs = {}

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_name_or_path,
            use_safetensors=pretrained_model_name_or_path.endswith(".safetensors"),
            revision=revision,
            variant=variant,
            local_files_only=True,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )
    else:
        pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            **pipeline_init_kwargs,
        )

    scheduler_config = pipeline.scheduler.config
    if prediction_type:
        scheduler_config['prediction_type'] = prediction_type
    if rescale_betas_zero_snr is not None:
        scheduler_config['rescale_betas_zero_snr'] = rescale_betas_zero_snr
    pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if no_half_vae:
        print("No half vae")
        pipeline.vae = pipeline.vae.to(torch.float32)

    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline
