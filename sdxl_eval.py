import torch
import os
from absl import flags
from absl import app
from ml_collections import config_flags
from pathlib import Path
from modules import sdxl_eval_utils, sdxl_train_utils, log_utils
from modules.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline


@torch.no_grad()
def eval(argv):
    config = flags.FLAGS.config
    config.output_dir = log_utils.smart_path(os.path.dirname(config.output_dir), os.path.basename(config.output_dir))
    accelerator = sdxl_eval_utils.prepare_accelerator(config)

    is_main_process = accelerator.is_main_process
    local_process_index = accelerator.state.local_process_index
    num_processes = accelerator.state.num_processes

    logger = log_utils.get_logger("eval", disable=not is_main_process)

    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae_dtype = torch.float32 if config.no_half_vae else weight_dtype

    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_utils.load_target_model(config, accelerator, "sdxl", weight_dtype)

    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(config.tokenizer_cache_dir, config.max_token_length)

    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()
    text_encoder1.to(weight_dtype)
    text_encoder2.to(weight_dtype)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    text_encoder1.eval()
    text_encoder2.eval()

    if config.diffusers_xformers:
        sdxl_train_utils.set_diffusers_xformers_flag(vae, True)
    else:
        logger.print("Disable Diffusers' xformers")
        sdxl_train_utils.replace_unet_modules(unet, config.mem_eff_attn, config.xformers, config.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(config.xformers)

    sample_dir = Path(config.output_dir)

    gen_params = sdxl_eval_utils.load_params(config.sample_benchmark)
    sample_sampler = sdxl_eval_utils.prepare_sampler(config.sample_sampler)

    pipe = SdxlStableDiffusionLongPromptWeightingPipeline(
        unet=unet,
        text_encoder=[text_encoder1, text_encoder2],
        vae=vae,
        tokenizer=[tokenizer1, tokenizer2],
        scheduler=sample_sampler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=config.clip_skip,
    )
    pipe.to(accelerator.device)

    pbar = logger.tqdm(total=len(gen_params), desc='total')
    for idx, param in enumerate(gen_params):

        param = sdxl_eval_utils.patch_default_param(param, config.sample_params)
        param = sdxl_eval_utils.prepare_param(param)

        def save_latents_callback(i, t, latents):
            images = pipe.latents_to_image(latents)
            for b, image in enumerate(images):
                sample_path = sample_dir / f"{idx:04d}-{b}-t={t:.0f}-i={i}.png"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(sample_path)

        with accelerator.autocast():
            latents = pipe(
                prompt=[param["prompt"]]*param["batch_size"],
                negative_prompt=[param["negative_prompt"]]*param["batch_size"],
                num_inference_steps=param["steps"],
                guidance_scale=param["scale"],
                width=param["width"],
                height=param["height"],
                original_width=param["original_width"],
                original_height=param["original_height"],
                original_scale_factor=param["original_scale_factor"],
                num_images_per_prompt=param["batch_count"],
                callback=save_latents_callback if param["save_latents"] else None,
            )
        images = pipe.latents_to_image(latents)

        sample_dir.mkdir(parents=True, exist_ok=True)
        for b, image in enumerate(images):
            info_dict = {
                'prompt': param["prompt"],
                'negative_prompt': param["negative_prompt"],
                'sample_steps': param["steps"],
                'width': param["width"],
                'height': param["height"],
                'scale': param["scale"],
                'seed': param["seed"],
                'sampler': config.sample_sampler,
                'original_width': param["original_width"],
                'original_height': param["original_height"],
                'original_scale_factor': param["original_scale_factor"],
            }
            output_path = sample_dir / f"{idx:04d}-{b}.png"
            image.save(output_path, save_all=True, append_images=[image], **info_dict)
        pbar.update(1)

    pbar.close()
    del pipe
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    del accelerator


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    app.run(eval)
