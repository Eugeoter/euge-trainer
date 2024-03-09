import torch
import os
import re
import time
from accelerate import Accelerator
from PIL import Image

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def prepare_accelerator(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        cpu=args.cpu,
    )
    return accelerator


def prepare_sampler(sample_sampler):
    # schedulerを用意する
    sched_init_args = {}
    if sample_sampler == "ddim":
        from diffusers import DDIMScheduler
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        from diffusers import DDPMScheduler
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        from diffusers import PNDMScheduler
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        from diffusers import LMSDiscreteScheduler
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        from diffusers import EulerDiscreteScheduler
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        from diffusers import EulerAncestralDiscreteScheduler
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        from diffusers import DPMSolverMultistepScheduler
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        from diffusers import DPMSolverSinglestepScheduler
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        from diffusers import HeunDiscreteScheduler
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        from diffusers import KDPM2DiscreteScheduler
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        from diffusers import KDPM2AncestralDiscreteScheduler
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def prepare_gen_params(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        params = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif path.endswith(".toml"):
        import toml
        with open(path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        params = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif path.endswith(".json"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

    return params


def parse_prompt(prompt):
    if isinstance(prompt, dict):
        negative_prompt = prompt.get("negative_prompt")
        sample_steps = prompt.get("sample_steps", 30)
        width = prompt.get("width", 512)
        height = prompt.get("height", 512)
        scale = prompt.get("scale", 7.5)
        seed = prompt.get("seed")
        controlnet_image = prompt.get("controlnet_image")
        prompt = prompt.get("prompt")
    else:
        # prompt = prompt.strip()
        # if len(prompt) == 0 or prompt[0] == "#":
        #     continue

        # subset of gen_img_diffusers
        prompt_args = prompt.split(" --")
        prompt = prompt_args[0]
        negative_prompt = None
        sample_steps = 30
        width = height = 512
        scale = 7.5
        seed = None
        controlnet_image = None
        for parg in prompt_args:
            try:
                m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                if m:
                    width = int(m.group(1))
                    continue

                m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                if m:
                    height = int(m.group(1))
                    continue

                m = re.match(r"d (\d+)", parg, re.IGNORECASE)
                if m:
                    seed = int(m.group(1))
                    continue

                m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                if m:  # steps
                    sample_steps = max(1, min(1000, int(m.group(1))))
                    continue

                m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                if m:  # scale
                    scale = float(m.group(1))
                    continue

                m = re.match(r"n (.+)", parg, re.IGNORECASE)
                if m:  # negative prompt
                    negative_prompt = m.group(1)
                    continue

                m = re.match(r"cn (.+)", parg, re.IGNORECASE)
                if m:  # negative prompt
                    controlnet_image = m.group(1)
                    continue

            except ValueError as ex:
                print(f"Exception in parsing / 解析エラー: {parg}")
                print(ex)

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)

    return prompt, negative_prompt, sample_steps, width, height, scale, seed, controlnet_image


@torch.no_grad()
def sample_single(
    pipe,
    prompt,
    negative_prompt,
    height,
    width,
    sample_steps,
    scale,
    controlnet_image,
    controlnet,
    seed,
):
    latents = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=sample_steps,
        guidance_scale=scale,
        negative_prompt=negative_prompt,
        controlnet=controlnet,
        controlnet_image=controlnet_image,
        seed=seed,
    )

    image = pipe.latents_to_image(latents)[0]

    return image


# def sample_images(*args, **kwargs):
#     try:
#         from .sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
#         return sample_during_train(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
#     except Exception as e:
#         print(f"Error in sample_images: {e}")
#         return None


def sample_during_train(
    pipe_class,
    accelerator,
    args,
    epoch,
    steps,
    unet,
    text_encoder,
    vae,
    tokenizer,
    device,
    prompt_replacement=None,
    controlnet=None,
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    """
    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return
    if args.sample_every_n_epochs is not None:
        # sample_every_n_steps は無視する
        if epoch is None or epoch % args.sample_every_n_epochs != 0:
            return
    else:
        if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
            return

    print(f"\ngenerating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        print(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(device)

    # read prompts

    # with open(args.sample_prompts, "rt", encoding="utf-8") as f:
    #     prompts = f.readlines()

    gen_params = prepare_gen_params(args.sample_prompts)
    scheduler = prepare_sampler(args.sample_sampler)

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(device)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad():
        # with accelerator.autocast():
        for i, param in enumerate(gen_params):
            if not accelerator.is_main_process:
                continue

            prompt, negative_prompt, sample_steps, width, height, scale, seed, controlnet_image = parse_prompt(param)

            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

            if prompt_replacement is not None:
                prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
                if negative_prompt is not None:
                    negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

            height = max(64, height - height % 32)  # round to divisible by 8
            width = max(64, width - width % 32)  # round to divisible by 8
            print(f"prompt: {prompt}")
            print(f"negative_prompt: {negative_prompt}")
            print(f"height: {height}")
            print(f"width: {width}")
            print(f"sample_steps: {sample_steps}")
            print(f"scale: {scale}")
            with accelerator.autocast():
                latents = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=sample_steps,
                    guidance_scale=scale,
                    negative_prompt=negative_prompt,
                    controlnet=controlnet,
                    controlnet_image=controlnet_image,
                )

            image = pipeline.latents_to_image(latents)[0]

            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            seed_suffix = "" if seed is None else f"_{seed}"
            img_filename = (
                f"{'' if args.output_name is None else args.output_name + '_'}{ts_str}_{num_suffix}_{i:02d}{seed_suffix}.png"
            )

            image.save(os.path.join(save_dir, img_filename))

            # wandb有効時のみログを送信
            try:
                wandb_tracker = accelerator.get_tracker("wandb")
                try:
                    import wandb
                except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
                    raise ImportError("No wandb / wandb がインストールされていないようです")

                wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
            except:  # wandb 無効時
                pass

    # clear pipeline and cache to reduce vram usage
    del pipeline
    torch.cuda.empty_cache()

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)
