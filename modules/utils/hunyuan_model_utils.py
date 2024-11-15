import torch
import os
from waifuset import logging
from typing import Literal, Tuple

logger = logging.get_logger("model")

NOISE_SCHEDULES = {
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
}

PREDICT_TYPE = {
    "epsilon",
    "sample",
    "v_prediction",
}

# =======================================================

NEGATIVE_PROMPT = '错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，'

# =======================================================
TRT_MAX_BATCH_SIZE = 1
TRT_MAX_WIDTH = 1280
TRT_MAX_HEIGHT = 1280

# =======================================================
# Constants about models
# =======================================================

VAE_EMA_PATH = "ckpts/t2i/sdxl-vae-fp16-fix"
TOKENIZER = "ckpts/t2i/tokenizer"
TEXT_ENCODER = 'ckpts/t2i/clip_text_encoder'
T5_ENCODER = {
    'MT5': 'ckpts/t2i/mt5',
    'attention_mask': True,
    'layer_index': -1,
    'attention_pool': True,
    'torch_dtype': torch.float16,
    'learnable_replace': True
}

SAMPLER_FACTORY = {
    'ddpm': {
        'scheduler': 'DDPMScheduler',
        'name': 'DDPM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.03,
            'prediction_type': 'v_prediction',
        }
    },
    'ddim': {
        'scheduler': 'DDIMScheduler',
        'name': 'DDIM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.03,
            'prediction_type': 'v_prediction',
        }
    },
    'dpmms': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPMMS',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.03,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
            'solver_order': 2,
            'algorithm_type': 'dpmsolver++',
        }
    },
}


def load_models_from_hunyuan_official(
    pretrained_model_name_or_path,
    model_args,
    dtype_t5: torch.dtype = T5_ENCODER['torch_dtype'],
    input_size: Tuple[int, int] = (128, 128),
    model_type: Literal['DiT-g/2', 'DiT-XL/2'] = 'DiT-XL/2',
    max_token_length_t5=None,
    use_deepspeed=False,
    deepspeed_config=None,
    deepspeed_remote_device=None,
    deepspeed_zero_stage=3,
):
    from diffusers.models.autoencoders import AutoencoderKL
    from transformers import BertModel, BertTokenizer
    from ..models.hunyuan.modules.models_controlnext import HUNYUAN_DIT_MODELS
    from ..models.hunyuan.modules.text_encoder import MT5Embedder

    logger.print(f"load HunyuanDiT official pretrained models: {pretrained_model_name_or_path}")

    if use_deepspeed:
        import deepspeed
        with deepspeed.zero.Init(
            data_parallel_group=torch.distributed.group.WORLD,
            remote_device=deepspeed_remote_device,
            config_dict_or_path=deepspeed_config,
            mpu=None,
            enabled=deepspeed_zero_stage == 3
        ):
            transformer = HUNYUAN_DIT_MODELS[model_type](
                model_args,
                input_size=input_size,
                log_fn=logger.info,
            )
    else:
        transformer = HUNYUAN_DIT_MODELS[model_type](
            model_args,
            input_size=input_size,
            log_fn=logger.info,
        )

    nnet_path = os.path.join(pretrained_model_name_or_path, 'model', 'pytorch_model_module.pt')
    sd = torch.load(nnet_path, map_location=lambda storage, loc: storage)
    sd = sd['module'] if 'module' in sd.keys() else sd
    transformer.load_state_dict(sd, strict=True)

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="sdxl-vae-fp16-fix",
    )
    # Setup BERT text encoder
    text_encoder = BertModel.from_pretrained(
        pretrained_model_name_or_path,
        False,
        subfolder="clip_text_encoder",
        revision=None,
    )
    # Setup BERT tokenizer:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    # Setup T5 text encoder
    embedder_t5 = MT5Embedder(
        pretrained_model_name_or_path + '/mt5',
        torch_dtype=dtype_t5,
        max_length=max_token_length_t5
    )
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    models = {
        "nnet": transformer, "vae": vae,
        "text_encoder1": text_encoder, "tokenizer1": tokenizer,
        "text_encoder2": text_encoder_t5, "tokenizer2": tokenizer_t5,
    }

    return models


# def load_models_from_hunyuan_diffusers_state(pretrained_model_name_or_path, device="cpu", dtype=None, revision=None, variant=None, cache_dir=None, token=None, dropout_t5=True, max_retries=None):
#     # Diffusers model is loaded to CPU
#     logger.print(f"load diffusers pretrained models: {pretrained_model_name_or_path}")
#     from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline
#     retries = 0
#     while True:
#         try:
#             kwargs = {}
#             if dropout_t5:
#                 kwargs['text_encoder_2'] = None
#                 kwargs['tokenizer_2'] = None
#             pipe: HunyuanDiTPipeline = HunyuanDiTPipeline.from_pretrained(
#                 pretrained_model_name_or_path,
#                 revision=revision,
#                 variant=variant,
#                 token=token,
#                 torch_dtype=dtype,
#                 cache_dir=cache_dir,
#                 **kwargs,
#             ).to(device)
#             break
#         except model_utils.NETWORK_EXCEPTIONS:
#             if max_retries is not None and retries >= max_retries:
#                 raise
#             retries += 1
#             logger.warning(f"Connection error when downloading model `{pretrained_model_name_or_path}` from HuggingFace. Retrying...")
#             continue
#     text_encoder1, text_encoder2 = pipe.text_encoder, pipe.text_encoder_2
#     tokenizer1, tokenizer2 = pipe.tokenizer, pipe.tokenizer_2
#     scheduler = pipe.scheduler
#     vae = pipe.vae
#     transformer = pipe.transformer
#     del pipe
#     logger.print("U-Net converted to original U-Net")
#     return {
#         "nnet": transformer, "vae": vae,
#         "text_encoder1": text_encoder1, "text_encoder2": text_encoder2,
#         "tokenizer1": tokenizer1, "tokenizer2": tokenizer2,
#         "noise_scheduler": scheduler,
#     }


# def load_models_from_hunyuan_checkpoint(ckpt_path, device="cpu", dtype=None):
#     if model_utils.is_safetensors(ckpt_path):
#         from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline
#         pipe: HunyuanDiTPipeline = HunyuanDiTPipeline.from_single_file(
#             ckpt_path,
#             use_safetensors=True,
#             torch_dtype=dtype,
#         )
#         text_encoder1, text_encoder2 = pipe.text_encoder, pipe.text_encoder_2, pipe
#         tokenizer1, tokenizer2 = pipe.tokenizer, pipe.tokenizer_2
#         noise_scheduler = pipe.scheduler
#         vae = pipe.vae
#         transformer = pipe.transformer
#         del pipe
#         logger.print("U-Net converted to original U-Net")
#         return {
#             "nnet": transformer, "vae": vae,
#             "text_encoder1": text_encoder1, "text_encoder2": text_encoder2,
#             "tokenizer1": tokenizer1, "tokenizer2": tokenizer2,
#             "noise_scheduler": noise_scheduler,
#         }
#     else:
#         raise NotImplementedError


def get_sampler(sampler: Literal['ddpm', 'ddim', 'dpmms']):
    from diffusers import schedulers
    SAMPLER_FACTORY = {
        'ddpm': {
            'scheduler': 'DDPMScheduler',
            'name': 'DDPM',
            'kwargs': {
                'steps_offset': 1,
                'clip_sample': False,
                'clip_sample_range': 1.0,
                'beta_schedule': 'scaled_linear',
                'beta_start': 0.00085,
                'beta_end': 0.03,
                'prediction_type': 'v_prediction',
            }
        },
        'ddim': {
            'scheduler': 'DDIMScheduler',
            'name': 'DDIM',
            'kwargs': {
                'steps_offset': 1,
                'clip_sample': False,
                'clip_sample_range': 1.0,
                'beta_schedule': 'scaled_linear',
                'beta_start': 0.00085,
                'beta_end': 0.03,
                'prediction_type': 'v_prediction',
            }
        },
        'dpmms': {
            'scheduler': 'DPMSolverMultistepScheduler',
            'name': 'DPMMS',
            'kwargs': {
                'beta_schedule': 'scaled_linear',
                'beta_start': 0.00085,
                'beta_end': 0.03,
                'prediction_type': 'v_prediction',
                'trained_betas': None,
                'solver_order': 2,
                'algorithm_type': 'dpmsolver++',
            }
        },
    }

    kwargs = SAMPLER_FACTORY[sampler]['kwargs']
    scheduler = SAMPLER_FACTORY[sampler]['scheduler']

    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)

    return scheduler


def save_hunyuan_checkpoint(output_dir, model, text_encoder1=None, text_encoder2=None, ema=None, epoch=None, step=None):
    client_state = {}
    if step is not None:
        client_state['steps'] = step
    if epoch is not None:
        client_state['epoch'] = epoch
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    os.makedirs(output_dir, exist_ok=True)
    client_state = {k: str(v) for k, v in client_state.items()}
    try:
        from safetensors.torch import save_model
        save_model(model, os.path.join(output_dir, 'nnet.safetensors'), metadata=client_state)
        if text_encoder1 is not None:
            save_model(text_encoder1, os.path.join(output_dir, 'text_encoder1.safetensors'))
        if text_encoder2 is not None:
            save_model(text_encoder2, os.path.join(output_dir, 'text_encoder2.safetensors'))
        logger.info(f"Saved checkpoint to {output_dir}")
    except Exception as e:
        logger.error(f"Saved failed to {output_dir}. {type(e)}: {e}")
