import torch
from pathlib import Path
from tqdm import tqdm
from modules import sdxl_eval_utils, sdxl_train_utils, arg_utils
from modules.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline


@torch.no_grad()
def eval(args):
    accelerator = sdxl_eval_utils.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    is_main_process = accelerator.is_main_process
    local_process_index = accelerator.state.local_process_index
    num_processes = accelerator.state.num_processes

    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_utils.load_target_model(args, accelerator, "sdxl", weight_dtype)

    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(args.tokenizer_cache_dir, args.max_token_length)

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

    if args.diffusers_xformers:
        sdxl_train_utils.set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        sdxl_train_utils.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    output_dir = Path(args.output_dir)

    gen_params = sdxl_eval_utils.prepare_gen_params(args.benchmark_file)
    scheduler = sdxl_eval_utils.prepare_sampler(args.sample_sampler)

    pipe = SdxlStableDiffusionLongPromptWeightingPipeline(
        unet=unet,
        text_encoder=[text_encoder1, text_encoder2],
        vae=vae,
        tokenizer=[tokenizer1, tokenizer2],
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipe.to(accelerator.device)

    pbar = tqdm(total=len(gen_params), disable=not is_main_process, desc='total')
    for idx, param in enumerate(gen_params):
        prompt, negative_prompt, sample_steps, width, height, scale, seed, _ = sdxl_eval_utils.parse_prompt(param)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        height = max(64, height - height % 32)  # round to divisible by 8
        width = max(64, width - width % 32)  # round to divisible by 8
        # print(f"prompt: {prompt}")
        # print(f"negative_prompt: {negative_prompt}")
        # print(f"height: {height}")
        # print(f"width: {width}")
        # print(f"sample_steps: {sample_steps}")
        # print(f"scale: {scale}")
        with accelerator.autocast():
            latents = pipe(
                prompt=[prompt]*args.batch_size,
                height=height,
                width=width,
                num_images_per_prompt=args.num_samples_per_prompt,
                num_inference_steps=sample_steps,
                guidance_scale=scale,
                negative_prompt=negative_prompt,
                # controlnet=controlnet,
                # controlnet_image=controlnet_image,
            )
        images = pipe.latents_to_image(latents)

        output_dir.mkdir(parents=True, exist_ok=True)
        for b, image in enumerate(images):
            output_path = output_dir / f"{idx:04d}-{b}.png"
            image.save(output_path)
        pbar.update(1)

    pbar.close()
    del pipe
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    del accelerator


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    arg_utils.add_model_arguments(parser)
    arg_utils.add_eval_arguments(parser)

    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()
