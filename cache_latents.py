import torch
import gc
from modules.sdxl_dataset_utils import Dataset
from modules import model_utils, sdxl_train_utils, sdxl_dataset_utils, arg_utils, log_utils as logu


def cache_latents(
    args,
):
    accelerator = sdxl_train_utils.prepare_accelerator(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    latents_dtype = sdxl_train_utils.prepare_dtype(args)

    vae_dtype = torch.float32 if args.no_half_vae else latents_dtype
    if args.vae is None:
        raise ValueError("vae path is not specified")
    vae = model_utils.load_vae(args.vae, vae_dtype)
    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(args.tokenizer_cache_dir, args.max_token_length)

    print(f"prepare dataset...")
    dataset = sdxl_dataset_utils.Dataset(
        args=args,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        latents_dtype=latents_dtype,
        is_main_process=accelerator.is_main_process,
        num_processes=accelerator.num_processes,
        process_idx=accelerator.local_process_index,
    )

    with torch.no_grad():
        dataset.cache_latents(
            vae,
            vae_batch_size=args.vae_batch_size,
            cache_to_disk=args.cache_latents_to_disk,
            check_validity=args.check_cache_validity,
            cache_only=True,
            async_cache=args.async_cache,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logu.success(f"cached latents at process {accelerator.local_process_index+1}/{accelerator.num_processes}")
    accelerator.wait_for_everyone()
    del accelerator


if __name__ == "__main__":
    args = arg_utils.add_train_arguments()
    cache_latents(args)
