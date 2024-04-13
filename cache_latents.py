import torch
import gc
import os
from absl import flags
from absl import app
from ml_collections import config_flags
from accelerate import Accelerator
from modules import model_utils, sdxl_train_utils, sdxl_dataset_utils, log_utils


def cache_latents(
    argv,
):
    config = flags.FLAGS.config
    accelerator = accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        cpu=config.cpu,
    )
    logger = log_utils.get_logger("cache")
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    latents_dtype = torch.float32
    if config.mixed_precision == "fp16":
        latents_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        latents_dtype = torch.bfloat16

    vae_dtype = torch.float32 if config.no_half_vae else latents_dtype
    if config.vae is not None:
        vae = model_utils.load_vae(config.vae, vae_dtype)
    else:
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_utils.load_target_model(config, accelerator, "sdxl", vae_dtype)
        del text_encoder1, text_encoder2, unet, logit_scale, ckpt_info
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(config.tokenizer_cache_dir, config.max_token_length)

    logger.print(f"prepare dataset...")
    dataset = sdxl_dataset_utils.Dataset(
        config=config,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        latents_dtype=latents_dtype,
        is_main_process=accelerator.is_main_process,
        num_processes=accelerator.num_processes,
        process_idx=accelerator.local_process_index,
        cache_only=True,
    )

    with torch.no_grad():
        dataset.cache_latents(
            vae=vae,
            accelerator=accelerator,
            vae_batch_size=config.vae_batch_size,
            cache_to_disk=config.cache_latents_to_disk,
            check_validity=config.check_cache_validity,
            async_cache=config.async_cache,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.print(log_utils.green(f"cache latents finished at process {accelerator.local_process_index+1}/{accelerator.num_processes}"), disable=False)
    accelerator.wait_for_everyone()
    del accelerator


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    app.run(cache_latents)
