import torch
import os
import inspect
import wandb
import numpy as np
from PIL import Image
from pathlib import Path
from waifuset import logging
from . import dataset_utils, device_utils

logger = logging.get_logger("eval")

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

DEFAULT_SAMPLE_NAME = 'image'


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


def get_sampler(sample_sampler, **sampler_kwargs):
    sample_sampler = sample_sampler.lower().replace(' ', '_')
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

    sched_init_args.update(sampler_kwargs)
    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def load_params_from_file(path):
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


def load_params_from_dicts(dicts):
    params = []
    for dic in dicts:
        # logger.debug(f"param_dict: {dic}, type: {type(dic)}")
        p = {}
        if (prompt := dic.get("prompt", dic.get('caption', dic.get('tags')))) is not None:
            p["prompt"] = prompt
        if (condition_image := dic.get("condition_image")) is not None:
            p["condition_image"] = condition_image
        elif (condition_image_path := dic.get("condition_image_path")) is not None:
            p["condition_image_path"] = condition_image_path
        if (control_scale := dic.get("control_scale")) is not None:
            p["control_scale"] = control_scale
        if (sample_name := dic.get("sample_name")) is not None:
            p["sample_name"] = sample_name
        params.append(p)
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
    "target_image",
    "target_image_path",
    "condition_image",
    "condition_image_path",
    "control_scale",  # for ControlNeXt
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


def prepare_param(param):
    prompt = param.get('prompt', param.get('caption', param.get('tags')))
    if prompt is None:
        prompt = ""
        logger.warning(f"Prompt not found, set to empty string")

    height = param["height"]
    width = param["width"]
    height = max(64, height - height % 32)
    width = max(64, width - width % 32)

    if (condition_image := param.get('condition_image')) is not None:
        import io
        if isinstance(condition_image, Image.Image):
            pass
        elif isinstance(condition_image, dict):
            if (condition_image_bytes := condition_image.get('bytes')) is not None:
                condition_image = Image.open(io.BytesIO(condition_image_bytes))
            elif (condition_image_path := condition_image.get('path')) is not None:
                condition_image = Image.open(condition_image_path)
            else:
                raise ValueError(f"condition image not found")
    elif (condition_image_path := param.get('condition_image_path')) is not None:
        condition_image = Image.open(condition_image_path).convert('RGB')
    else:
        condition_image = None

    param["prompt"] = prompt
    param["height"] = height
    param["width"] = width
    param["generator"] = torch.Generator().manual_seed(param["seed"])
    param["condition_image"] = condition_image
    param["target_image"] = param.get("target_image")

    return param


def overlaid_image(image, conditioning_image):
    if image.size != conditioning_image.size:
        raise ValueError(f"image and conditioning_image should have the same size, but image size is {image.size} and conditioning_image size is {conditioning_image.size}")
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
    benchmark,
    default_params,
    epoch,
    steps,
    device,
    wandb_run=None,
):
    logger.info(f"\ngenerating sample images at step: {steps}")

    device_utils.clean_memory_on_device(device)
    use_tmp_vae_device = False
    if hasattr(pipeline, "vae") and (vae := pipeline.vae) is not None:
        orig_vae_device = vae.device
        vae.to(device)
        use_tmp_vae_device = True

    # read prompts
    if isinstance(benchmark, (str, Path)):
        gen_params = load_params_from_file(benchmark)
    elif isinstance(benchmark, list) and all(isinstance(d, dict) for d in benchmark):
        gen_params = benchmark
    else:
        raise ValueError("benchmark should be a path to a file or a list of dictionaries")
    pipeline.to(device)
    acceptable_param_names = inspect.signature(pipeline.__call__).parameters.keys()

    # sample_dir = os.path.join(config.output_dir, config.output_subdir.samples, f"epoch_{epoch}" if epoch is not None else f"step_{steps}")
    os.makedirs(sample_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad():
        for i, param in enumerate(gen_params):
            if not accelerator.is_main_process:
                continue

            sample_name = param.pop("sample_name", DEFAULT_SAMPLE_NAME)

            param = patch_default_param(param, default_params)
            param = prepare_param(param)
            target_image = param.get("target_image")
            condition_image = param.get("condition_image")

            has_image_condition = condition_image is not None
            has_target_image = target_image is not None
            logger.debug(f"is_condition: {has_image_condition}")

            logger.info(f"sample_{i}:")
            logger.info(f"  sample_name: {sample_name}", no_prefix=True)
            logger.info(f"  prompt: {param['prompt']}", no_prefix=True)
            logger.info(f"  negative_prompt: {param['negative_prompt']}", no_prefix=True)
            logger.info(f"  seed: {param['seed']}", no_prefix=True)

            if has_image_condition:
                logger.info(f"  image condition provided", no_prefix=True)

                if 'control_scale' in acceptable_param_names:  # for ControlNeXt
                    if 'control_scale' in param:
                        logger.info(f"  control_scale: {param['control_scale']}", no_prefix=True)
                    else:
                        logger.info(f"  control_scale: 1.0 (default)", no_prefix=True)

            # logger.debug(f"default_params: {default_params}")
            # logger.debug(f"param: {param}")

            # def save_latents_callback(step, timestep, latents):
            #     images = pipeline.latents_to_image(latents)
            #     for j, image in enumerate(images):
            #         num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            #         sample_path = sample_dir / f"latents-{num_suffix}-{i}-{j}-timestep{timestep:.0f}-step{step}.png"  # FIXME: idx is not defined
            #         sample_path.parent.mkdir(parents=True, exist_ok=True)
            #         image.save(sample_path)

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
                if has_image_condition:
                    if "image" in acceptable_param_names:
                        if 'mask_image' in acceptable_param_names:  # for inpainting pipeline
                            assert isinstance(target_image, (Image.Image, np.ndarray, torch.Tensor)), f"target_image should be Image.Image, np.ndarray or torch.Tensor, but got {type(target_image)}"
                            pipeline_input["image"] = target_image
                            pipeline_input["mask_image"] = condition_image
                            logger.info(f"  inpainting condition provided", no_prefix=True)
                        else:
                            pipeline_input["image"] = condition_image
                            logger.info(f"  image condition provided", no_prefix=True)
                    elif "controlnet_image" in acceptable_param_names:
                        pipeline_input["controlnet_image"] = condition_image
                        logger.info(f"  controlnet condition provided", no_prefix=True)
                    else:
                        logger.warning(f"condition_image is provided but not used in the pipeline")
                    if 'controlnet_scale' in acceptable_param_names and 'control_scale' in param:
                        pipeline_input['controlnet_scale'] = param["control_scale"]
                    pipeline_input['width'] = condition_image.width // 8 * 8
                    pipeline_input['height'] = condition_image.height // 8 * 8
                else:
                    if 'controlnet' in pipeline.__class__.__name__.lower():
                        logger.warning(f"Pipeline {pipeline.__class__.__name__} looks like a controlnet-related pipeline but condition condition is not provided")
                    if 'controlnext' in pipeline.__class__.__name__.lower():
                        logger.warning(f"Pipeline {pipeline.__class__.__name__} looks like a controlnext-related pipeline but condition condition is not provided")
                    if 'inpaint' in pipeline.__class__.__name__.lower():
                        logger.warning(f"Pipeline {pipeline.__class__.__name__} looks like a inpainting-related pipeline but condition condition is not provided")

                output = pipeline(**pipeline_input)

            # save image
            if isinstance(output, torch.Tensor):
                images = pipeline.latents_to_image(output)
            else:
                images = output.images
            if has_image_condition:
                condition_image: Image.Image = condition_image
                condition_image = condition_image.convert('RGB')
                condition_image = dataset_utils.resize_if_needed(condition_image, (pipeline_input["width"], pipeline_input["height"]))
                images_overlaid = []
            for j, image in enumerate(images):
                # ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                img_filename = f"{sample_name}-{num_suffix}-{i}-{j}.png"
                image.save(os.path.join(sample_dir, img_filename))
                if wandb_run is not None:
                    wandb_run.log({img_filename: wandb.Image(image)}, step=steps)
                if has_image_condition:
                    condition_image.save(os.path.join(sample_dir, f"{sample_name}_condition-{num_suffix}-{i}-{j}.png"))
                    if wandb_run is not None:
                        wandb_run.log({f"condition_image-{num_suffix}-{i}-{j}.png": wandb.Image(condition_image)}, step=steps)

                    image_overlaid = overlaid_image(image, condition_image)
                    overlaid_filename = f"{sample_name}_overlaid-{num_suffix}-{i}-{j}.png"
                    image_overlaid.save(os.path.join(sample_dir, overlaid_filename))
                    if wandb_run is not None:
                        wandb_run.log({overlaid_filename: wandb.Image(image_overlaid)}, step=steps)
                    images_overlaid.append(image_overlaid)
                if has_target_image:
                    target_image.save(os.path.join(sample_dir, f"{sample_name}_target-{num_suffix}-{i}-{j}.png"))
                    if wandb_run is not None:
                        wandb_run.log({f"target_image-{num_suffix}-{i}-{j}.png": wandb.Image(target_image)}, step=steps)

            if len(images) > 1:
                if has_image_condition:
                    images_grid = concat_images([condition_image] + images)
                else:
                    images_grid = concat_images(images)
                grid_filename = f"{sample_name}_grid-{num_suffix}-{i}.png"
                images_grid.save(os.path.join(sample_dir, grid_filename))
                if wandb_run is not None:
                    wandb_run.log({grid_filename: wandb.Image(images_grid)}, step=steps)
                if has_image_condition:
                    images_overlaid_grid = concat_images([condition_image] + images_overlaid)
                    overlaid_grid_filename = f"{sample_name}_overlaid_grid-{num_suffix}-{i}.png"
                    images_overlaid_grid.save(os.path.join(sample_dir, overlaid_grid_filename))
                    if wandb_run is not None:
                        wandb_run.log({overlaid_grid_filename: wandb.Image(images_overlaid_grid)}, step=steps)

    # clear pipeline and cache to reduce vram usage
    del pipeline
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    if use_tmp_vae_device:
        vae.to(orig_vae_device)
    device_utils.clean_memory_on_device(device)
