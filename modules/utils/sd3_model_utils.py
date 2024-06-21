import torch
import os
import copy
from diffusers import SD3Transformer2DModel
from transformers import PretrainedConfig
from . import model_utils, log_utils

logger = log_utils.get_logger("model")

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
TOKENIZER3_PATH = "google/flan-t5-xxl"


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = None
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, revision=revision, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_sd3_tokenizers(tokenizer_cache_dir=None, max_token_length=77):
    name_or_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH, TOKENIZER3_PATH]
    tokenizers = []
    for i, name_or_path in enumerate(name_or_paths):
        tokenizer_class = import_model_class_from_model_name_or_path(name_or_path, revision=None)
        tokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, name_or_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.print(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = tokenizer_class.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = tokenizer_class.from_pretrained(name_or_path)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokenizers.append(tokenizer)

    if max_token_length is not None:
        logger.print(f"update token length: {max_token_length}")

    return tokenizers


def load_models_from_stable_diffusion_checkpoint(ckpt_path, device="cpu", dtype=None, nnet_class=SD3Transformer2DModel):
    if model_utils.is_safetensors(ckpt_path):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_single_file(
            ckpt_path, use_safetensors=True,
            torch_dtype=dtype,
        )
        text_encoder1, text_encoder2, text_encoder3 = pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
        tokenizer1, tokenizer2, tokenizer3 = pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
        noise_scheduler = pipe.scheduler
        vae = pipe.vae
        transformer = pipe.transformer
        del pipe
        logger.print("U-Net converted to original U-Net")
        return {
            "nnet": transformer, "vae": vae,
            "text_encoder1": text_encoder1, "text_encoder2": text_encoder2, "text_encoder3": text_encoder3,
            "tokenizer1": tokenizer1, "tokenizer2": tokenizer2, "tokenizer3": tokenizer3,
            "noise_scheduler": noise_scheduler,
        }
    else:
        raise NotImplementedError


def load_models_from_stable_diffusion_diffusers_state(pretrained_model_name_or_path, device="cpu", dtype=None, revision=None, variant=None, cache_dir=None, token=None, nnet_class=SD3Transformer2DModel, dropout_t5=True, max_retries=None):
    # Diffusers model is loaded to CPU
    logger.print(f"load diffusers pretrained models: {pretrained_model_name_or_path}")
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
    retries = 0
    while True:
        try:
            kwargs = {}
            if dropout_t5:
                kwargs['text_encoder_3'] = None
                kwargs['tokenizer_3'] = None
            if nnet_class is not SD3Transformer2DModel:  # load nnet
                transformer = nnet_class.from_pretrained(
                    pretrained_model_name_or_path,
                    revision=revision,
                    variant=variant,
                    token=token,
                    torch_dtype=dtype,
                    cache_dir=cache_dir,
                    subfolder="transformer",
                )
                kwargs['transformer'] = transformer
            pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                variant=variant,
                token=token,
                torch_dtype=dtype,
                cache_dir=cache_dir,
                **kwargs,
            ).to(device)
            break
        except model_utils.NETWORK_EXCEPTIONS:
            if max_retries is not None and retries >= max_retries:
                raise
            retries += 1
            logger.print(log_utils.yellow(f"Connection error when downloading model `{pretrained_model_name_or_path}` from HuggingFace. Retrying..."))
            continue
    text_encoder1, text_encoder2, text_encoder3 = pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    tokenizer1, tokenizer2, tokenizer3 = pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
    scheduler = pipe.scheduler
    vae = pipe.vae
    transformer = pipe.transformer
    del pipe
    logger.print("U-Net converted to original U-Net")
    return {
        "nnet": transformer, "vae": vae,
        "text_encoder1": text_encoder1, "text_encoder2": text_encoder2, "text_encoder3": text_encoder3,
        "tokenizer1": tokenizer1, "tokenizer2": tokenizer2, "tokenizer3": tokenizer3,
        "noise_scheduler": scheduler,
    }


def load_models_from_huggingface(pretrained_model_name_or_path, revision=None, device="cpu", dtype=None, variant=None, cache_dir=None, token=None, nnet_class=SD3Transformer2DModel, max_retries=None):
    from transformers import CLIPTokenizer, T5TokenizerFast
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler = copy.deepcopy(noise_scheduler)

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_3", revision=revision, variant=variant
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant
    )

    return {
        "nnet": transformer, "vae": vae,
        "text_encoder1": text_encoder_one, "text_encoder2": text_encoder_two, "text_encoder3": text_encoder_three,
        "tokenizer1": tokenizer_one, "tokenizer2": tokenizer_two, "tokenizer3": tokenizer_three,
        "noise_scheduler": noise_scheduler,
    }
