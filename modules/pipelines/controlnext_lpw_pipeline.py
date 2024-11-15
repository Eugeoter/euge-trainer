from diffusers.image_processor import VaeImageProcessor
from ..models.sd15.controlnext import ControlNeXtModel
from .lpw_pipeline import *


class ControlNeXtLPWStableDiffusionPipeline(StableDiffusionLongPromptWeightingPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnext: ControlNeXtModel,
        scheduler: SchedulerMixin,
        # clip_skip: int,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        image_encoder: CLIPVisionModelWithProjection = None,
        clip_skip: int = 1,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            image_encoder=image_encoder,
        )

        self.register_modules(controlnext=controlnext)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.custom_clip_skip = clip_skip
        self.__init__additional__()

    def prepare_control_image(
        self,
        controlnet_image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        controlnet_image = self.control_image_processor.preprocess(controlnet_image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = controlnet_image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        controlnet_image = controlnet_image.repeat_interleave(repeat_by, dim=0)

        controlnet_image = controlnet_image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            controlnet_image = torch.cat([controlnet_image] * 2)

        return controlnet_image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        # controlnet=None,
        controlnet_image=None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            controlnet (`diffusers.ControlNetModel`, *optional*):
                A controlnet model to be used for the inference. If not provided, controlnet will be disabled.
            controlnet_image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch, to be used as the starting point for the controlnet
                inference.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if self.controlnext is not None and controlnet_image is None:
            raise ValueError("controlnet_image must be provided if controlnet is not None.")

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
        )
        dtype = text_embeddings.dtype

        # 4. Preprocess image and mask
        if isinstance(image, PIL.Image.Image):
            image = preprocess_image(image)
        if image is not None:
            image = image.to(device=self.device, dtype=dtype)
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = preprocess_mask(mask_image, self.vae_scale_factor)
        if mask_image is not None:
            mask = mask_image.to(device=self.device, dtype=dtype)
            mask = torch.cat([mask] * batch_size * num_images_per_prompt)
        else:
            mask = None

        if controlnet_image is not None:
            controlnet_image = self.prepare_control_image(
                controlnet_image=controlnet_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.controlnext.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
            height, width = controlnet_image.shape[-2:]

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, image is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, init_latents_orig, noise = self.prepare_latents(
            image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            unet_additional_args = {}
            if self.controlnext is not None:
                controlnet_output = self.controlnext(
                    controlnet_image,
                    t,
                )
                unet_additional_args["mid_block_additional_residual"] = controlnet_output

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, **unet_additional_args).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None

        return latents
