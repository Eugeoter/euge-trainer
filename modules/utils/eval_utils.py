import torch
import os
import numpy as np
from PIL import Image
from . import log_utils

logger = log_utils.get_logger("eval")

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


ALL_SAMPLERS = [
    "ddim",
    "ddpm",
    "pndm",
    "lms",
    "k_lms",
    "euler",
    "k_euler",
    "euler_a",
    "k_euler_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "heun",
    "dpm_2",
    "k_dpm_2",
    "dpm_2_a",
    "k_dpm_2_a",
]


def get_sampler(sample_sampler):
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
        # logger.print("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def load_params(path):
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


GEN_PARAMS = [
    "prompt",
    "negative_prompt",
    "batch_size",
    "batch_count",
    "steps",
    "width",
    "height",
    "scale",
    "seed",
    # "original_width",
    # "original_height",
    # "original_scale_factor",
    "control_image",
    "save_latents",
]


def patch_default_param(param, default_params):
    r"""
    Patch unassigned parameters in params with args
    """
    for key in GEN_PARAMS:
        if key not in param:
            if key in default_params:
                param[key] = default_params[key]
    return param


def set_seed(seed):
    if seed is None:
        seed = torch.randint(0, 2 ** 32, (1,)).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def prepare_param(param):
    height = param["height"]
    width = param["width"]
    height = max(64, height - height % 32)
    width = max(64, width - width % 32)
    seed = set_seed(param["seed"])

    control_image = Image.open(param["control_image"]).convert("RGB") if param.get("control_image") is not None else None
    param["height"] = height
    param["width"] = width
    param["seed"] = seed
    param["control_image"] = control_image

    return param


def overlaid_image(image, conditioning_image):
    image = image.convert("RGBA")
    conditioning_image = conditioning_image.convert("RGBA")
    new_img = Image.blend(image, conditioning_image, alpha=0.5)
    new_img = new_img.convert("RGB")
    return new_img


def concat_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def sample_during_train(
    pipeline,
    accelerator,
    sample_dir,
    benchmark_file,
    default_params,
    epoch,
    steps,
    device,
):
    logger.print(f"\ngenerating sample images at step: {steps}")

    use_tmp_vae_device = False
    if hasattr(pipeline, "vae") and (vae := pipeline.vae) is not None:
        orig_vae_device = vae.device
        vae.to(device)
        use_tmp_vae_device = True

    # read prompts
    gen_params = load_params(benchmark_file)
    pipeline.to(device)

    # sample_dir = os.path.join(config.output_dir, config.output_subdir.samples, f"epoch_{epoch}" if epoch is not None else f"step_{steps}")
    os.makedirs(sample_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad():
        for i, param in enumerate(gen_params):
            if not accelerator.is_main_process:
                continue

            param = patch_default_param(param, default_params)
            param = prepare_param(param)

            logger.print(f"sample_{i}:")
            logger.print(f"  prompt: {param['prompt']}", no_prefix=True)
            logger.print(f"  negative_prompt: {param['negative_prompt']}", no_prefix=True)
            logger.print(f"  seed: {param['seed']}", no_prefix=True)

            def save_latents_callback(step, timestep, latents):
                images = pipeline.latents_to_image(latents)
                for j, image in enumerate(images):
                    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                    sample_path = sample_dir / f"latents-{num_suffix}-{i}-{j}-timestep{timestep:.0f}-step{step}.png"  # FIXME: idx is not defined
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(sample_path)

            with accelerator.autocast():
                pipeline_input = dict(
                    prompt=[param["prompt"]]*param["batch_size"],
                    negative_prompt=[param["negative_prompt"]]*param["batch_size"],
                    num_inference_steps=param["steps"],
                    guidance_scale=param["scale"],
                    width=param["width"],
                    height=param["height"],
                    # original_width=param["original_width"],
                    # original_height=param["original_height"],
                    # original_scale_factor=param["original_scale_factor"],
                    num_images_per_prompt=param["batch_count"],
                    # callback_on_step_end_tensor_inputs=save_latents_callback if param["save_latents"] else None,
                )
                if param["control_image"] is not None:
                    pipeline_input["controlnet_image"] = param["control_image"]
                output = pipeline(**pipeline_input)

            # save image
            if isinstance(output, torch.Tensor):
                images = pipeline.latents_to_image(output)
            else:
                images = output.images
            has_controlnet = param.get("control_image") is not None
            if has_controlnet:
                control_image: Image.Image = param["control_image"]
                control_image = control_image.convert('RGB').resize((param["width"], param["height"]), resample=Image.LANCZOS)
                images_overlaid = []
            for j, image in enumerate(images):
                # ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                img_filename = f"image-{num_suffix}-{i}-{j}.png"
                image.save(os.path.join(sample_dir, img_filename))
                if has_controlnet:
                    image_overlaid = overlaid_image(image, control_image)
                    overlaid_filename = f"image_overlaid-{num_suffix}-{i}-{j}.png"
                    image_overlaid.save(os.path.join(sample_dir, overlaid_filename))
                    images_overlaid.append(image_overlaid)

            if has_controlnet:
                images_grid = concat_images([control_image] + images)
            else:
                images_grid = concat_images(images)
            grid_filename = f"images_grid-{num_suffix}-{i}.png"
            images_grid.save(os.path.join(sample_dir, grid_filename))
            if has_controlnet:
                images_overlaid_grid = concat_images([control_image] + images_overlaid)
                overlaid_grid_filename = f"images_overlaid_grid-{num_suffix}-{i}.png"
                images_overlaid_grid.save(os.path.join(sample_dir, overlaid_grid_filename))

            try:
                wandb_tracker = accelerator.get_tracker("wandb")
                try:
                    import wandb
                except ImportError:
                    raise ImportError("No wandb installed. Please install wandb to use this feature.")
                wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
            except:
                pass

    # clear pipeline and cache to reduce vram usage
    del pipeline
    torch.cuda.empty_cache()

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    if use_tmp_vae_device:
        vae.to(orig_vae_device)
