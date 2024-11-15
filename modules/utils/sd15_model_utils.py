# v1: split from train_db_fixed.py.
# v2: support safetensors

import math
import os
import json
import struct
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
import diffusers
import huggingface_hub.utils
import requests
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline  # , UNet2DConditionModel
from transformers import CLIPTokenizer
from safetensors.torch import load_file, save_file
from typing import Optional, Any, Dict
from waifuset import logging
from ..models.sd15.nnet import UNet2DConditionModel

logger = logging.get_logger("model")

# DiffUsers版StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64  # fixed from old invalid value `32`
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8
# UNET_PARAMS_USE_LINEAR_PROJECTION = False

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024
# V2_UNET_PARAMS_USE_LINEAR_PROJECTION = True

# Diffusersの設定を読み込むための参照モデル
DIFFUSERS_REF_MODEL_ID_V1 = "runwayml/stable-diffusion-v1-5"
DIFFUSERS_REF_MODEL_ID_V2 = "stabilityai/stable-diffusion-2-1"

TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

NETWORK_EXCEPTIONS = (huggingface_hub.utils.HfHubHTTPError, ConnectionError, requests.exceptions.HTTPError, requests.exceptions.ReadTimeout)


def load_sd15_tokenizer(name_or_path: str, subfolder=None, cache_dir: Optional[str] = None, max_token_length: Optional[int] = None):
    logger.print("prepare tokenizer")
    tokenizer: CLIPTokenizer = None
    if cache_dir:
        local_tokenizer_path = os.path.join(cache_dir, name_or_path.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            logger.print(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)  # same for v1 and v2

    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(name_or_path, subfolder=subfolder)  # same for v1 and v2

    if max_token_length is not None:
        logger.print(f"update token length: {max_token_length}")

    if cache_dir and not os.path.exists(local_tokenizer_path):
        logger.print(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


def load_tokenizer(
    model_class: Any, model_id: str, subfolder: Optional[str] = None, tokenizer_cache_dir: Optional[str] = None
) -> Any:
    tokenizer = None
    if tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

    if tokenizer is None:
        tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

    if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
        logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


def is_safetensors(path):
    return os.path.splitext(path)[1].lower() == ".safetensors"


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    # does not support metadata loading
    def __init__(self, filename):
        self.filename = filename
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # # convert to float16 if float8 is not supported
            # print(f"Warning: {dtype_str} is not supported in this PyTorch version. Converting to float16.")
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def load_checkpoint_with_text_encoder_conversion(ckpt_path, device="cpu"):
    # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
    TEXT_ENCODER_KEY_REPLACEMENTS = [
        ("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings."),
        ("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder."),
        ("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm."),
    ]

    if is_safetensors(ckpt_path):
        checkpoint = None
        state_dict = load_file(ckpt_path)  # , device) # may causes error
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None

    key_reps = []
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from):]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict


def load_model_from_stable_diffusion_state_dict(state_dict, device="cpu", dtype=None, v2=False, nnet_class=UNet2DConditionModel, unet_use_linear_projection_in_v2=True, strict=True):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(v2, unet_use_linear_projection_in_v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

    unet = nnet_class(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint, strict=strict)
    logger.print("loading u-net:", info)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

    vae = AutoencoderKL(**vae_config).to(device)
    info = vae.load_state_dict(converted_vae_checkpoint, strict=strict)
    logger.print("loading vae:", info)

    # convert text_model
    if v2:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=23,
            num_attention_heads=16,
            max_position_embeddings=77,
            hidden_act="gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=512,
            torch_dtype="float32",
            transformers_version="4.25.0.dev0",
        )
        text_encoder = CLIPTextModel._from_config(cfg)
        info = text_encoder.load_state_dict(converted_text_encoder_checkpoint, strict=strict)
    else:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)

        # logging.set_verbosity_error()  # don't show annoying warning
        # text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # logging.set_verbosity_warning()
        # logger.print(f"config: {text_model.config}")
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=768,
            torch_dtype="float32",
        )
        text_encoder = CLIPTextModel._from_config(cfg)
        info = text_encoder.load_state_dict(converted_text_encoder_checkpoint, strict=strict)
    logger.print("loading text encoder:", info)
    return {"nnet": unet, "vae": vae, "text_encoder": text_encoder}

# TODO dtype指定の動作が怪しいので確認する text_encoderを指定形式で作れるか未確認


def load_models_from_stable_diffusion_checkpoint(ckpt_path, device="cpu", dtype=None, v2=False, nnet_class=UNet2DConditionModel, unet_use_linear_projection_in_v2=True, strict=True):
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path, device)

    return load_model_from_stable_diffusion_state_dict(state_dict, device, dtype, v2, nnet_class, unet_use_linear_projection_in_v2, strict=strict)


def load_models_from_stable_diffusion_diffusers_state(version, device='cpu', dtype=None, nnet_class=UNet2DConditionModel, cache_dir=None, token=None, max_retries=None):
    # Diffusers model is loaded to CPU
    logger.print(f"load diffusers pretrained models: {version}")
    while True:
        try:
            unet = nnet_class.from_pretrained(version, subfolder='unet', cache_dir=cache_dir, token=token)
            break
        except NETWORK_EXCEPTIONS:
            if max_retries is not None and retries >= max_retries:
                raise
            retries += 1
            logger.print(logging.yellow(f"connection error when downloading model `{version}` from HuggingFace. Retrying..."))
            continue

    from diffusers import StableDiffusionPipeline
    retries = 0
    while True:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                version,
                unet=unet,
                tokenizer=None,
                cache_dir=cache_dir,
                token=token,
                safety_checker=None
            )
            break
        except NETWORK_EXCEPTIONS:
            if max_retries is not None and retries >= max_retries:
                raise
            retries += 1
            logger.print(logging.yellow(f"connection error when downloading model `{version}` from HuggingFace. Retrying..."))
            continue
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    del pipe

    # Diffusers U-Net to original U-Net
    # TODO *.ckpt/*.safetensorsのv2と同じ形式にここで変換すると良さそう
    # logger.print(f"unet config: {unet.config}")
    # original_unet = nnet_class(
    #     unet.config.sample_size,
    #     unet.config.attention_head_dim,
    #     unet.config.cross_attention_dim,
    #     unet.config.use_linear_projection,
    #     unet.config.upcast_attention,
    # )
    # original_unet.load_state_dict(unet.state_dict())
    # unet = original_unet
    # logger.print("U-Net converted to original U-Net")
    return {"nnet": unet, "vae": vae, "text_encoder": text_encoder}


def convert_models_to_stable_diffusion_checkpoint_state_dict(
    text_encoder, unet, ckpt_path=None, save_dtype=None, vae=None, v2=False, strict=False
):
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            assert not strict or key in state_dict, f"Illegal key in save SD: {key}"
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
    update_sd("model.diffusion_model.", unet_state_dict)

    # Convert the text encoder model
    if v2:
        make_dummy = ckpt_path is None  # 参照元のcheckpointがない場合は最後の層を前の層から複製して作るなどダミーの重みを入れる
        text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict(), make_dummy)
        update_sd("cond_stage_model.model.", text_enc_dict)
    else:
        text_enc_dict = text_encoder.state_dict()
        update_sd("cond_stage_model.transformer.", text_enc_dict)

    # Convert the VAE
    if vae is not None:
        vae_dict = convert_vae_state_dict(vae.state_dict())
        update_sd("first_stage_model.", vae_dict)

    return state_dict


def save_stable_diffusion_checkpoint(
    output_file, text_encoder, unet, ckpt_path, epochs, steps, metadata, save_dtype=None, vae=None, v2=False, use_mem_eff_save=False,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if ckpt_path is not None:
        # epoch/stepを参照する。またVAEがメモリ上にないときなど、もう一度VAEを含めて読み込む
        checkpoint, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
        if checkpoint is None:  # safetensors または state_dictのckpt
            checkpoint = {}
            strict = False
        else:
            strict = True
        if "state_dict" in state_dict:
            del state_dict["state_dict"]
    else:
        # 新しく作る
        assert vae is not None, "VAE is required to save a checkpoint without a given checkpoint"
        checkpoint = {}
        state_dict = {}
        strict = False

    state_dict = convert_models_to_stable_diffusion_checkpoint_state_dict(
        text_encoder, unet, ckpt_path=ckpt_path, save_dtype=save_dtype, vae=vae, v2=v2, strict=strict
    )

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # epoch and global_step are sometimes not int
    try:
        if "epoch" in checkpoint:
            epochs += checkpoint["epoch"]
        if "global_step" in checkpoint:
            steps += checkpoint["global_step"]
    except:
        pass

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if is_safetensors(output_file):
        # TODO Tensor以外のdictの値を削除したほうがいいか
        if use_mem_eff_save:
            mem_eff_save_file(state_dict, output_file, metadata)
        else:
            save_file(state_dict, output_file, metadata)

    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(output_dir, text_encoder, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False, v2=False):
    os.makedirs(output_dir, exist_ok=True)
    if pretrained_model_name_or_path is None:
        # load default settings for v1/v2
        if v2:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V2
        else:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V1

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # original U-Net cannot be saved, so we need to convert it to the Diffusers version
    # TODO this consumes a lot of memory
    diffusers_unet = diffusers.UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    diffusers_unet.load_state_dict(unet.state_dict())

    pipeline = StableDiffusionPipeline(
        unet=diffusers_unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)

# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーして修正している（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        if diffusers.__version__ < "0.17.0":
            new_item = new_item.replace("q.weight", "query.weight")
            new_item = new_item.replace("q.bias", "query.bias")

            new_item = new_item.replace("k.weight", "key.weight")
            new_item = new_item.replace("k.bias", "key.bias")

            new_item = new_item.replace("v.weight", "value.weight")
            new_item = new_item.replace("v.bias", "value.bias")

            new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
            new_item = new_item.replace("proj_out.bias", "proj_attn.bias")
        else:
            new_item = new_item.replace("q.weight", "to_q.weight")
            new_item = new_item.replace("q.bias", "to_q.bias")

            new_item = new_item.replace("k.weight", "to_k.weight")
            new_item = new_item.replace("k.bias", "to_k.bias")

            new_item = new_item.replace("v.weight", "to_v.weight")
            new_item = new_item.replace("v.bias", "to_v.bias")

            new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
            new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        reshaping = False
        if diffusers.__version__ < "0.17.0":
            if "proj_attn.weight" in new_path:
                reshaping = True
        else:
            if ".attentions." in new_path and ".0.to_" in new_path and old_checkpoint[path["old"]].ndim > 2:
                reshaping = True

        if reshaping:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key] for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}." in key] for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key] for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

            # オリジナル：
            # if ["conv.weight", "conv.bias"] in output_block_list.values():
            #   index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])

            # biasとweightの順番に依存しないようにする：もっといいやり方がありそうだが
            for l in output_block_list.values():
                l.sort()

            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    # SDのv2では1*1のconv2dがlinearに変わっている
    # 誤って Diffusers 側を conv2d のままにしてしまったので、変換必要
    if v2 and not config.get("use_linear_projection", False):
        linear_transformer_to_conv(new_checkpoint)

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)
    # if len(vae_state_dict) == 0:
    #   # 渡されたcheckpointは.ckptから読み込んだcheckpointではなくvaeのstate_dict
    #   vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)}

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)}

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def create_unet_diffusers_config(v2, use_linear_projection_in_v2=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # unet_params = original_config.model.params.unet_config.params

    block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=UNET_PARAMS_IMAGE_SIZE,
        in_channels=UNET_PARAMS_IN_CHANNELS,
        out_channels=UNET_PARAMS_OUT_CHANNELS,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
        cross_attention_dim=UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
        attention_head_dim=UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
        # use_linear_projection=UNET_PARAMS_USE_LINEAR_PROJECTION if not v2 else V2_UNET_PARAMS_USE_LINEAR_PROJECTION,
    )
    if v2 and use_linear_projection_in_v2:
        config["use_linear_projection"] = True

    return config


def create_vae_diffusers_config():
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # vae_params = original_config.model.params.first_stage_config.params.ddconfig
    # _ = original_config.model.params.first_stage_config.params.embed_dim
    block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=VAE_PARAMS_RESOLUTION,
        in_channels=VAE_PARAMS_IN_CHANNELS,
        out_channels=VAE_PARAMS_OUT_CH,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=VAE_PARAMS_Z_CHANNELS,
        layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
    )
    return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]

    # support checkpoint without position_ids (invalid checkpoint)
    if "text_model.embeddings.position_ids" not in text_model_dict:
        text_model_dict["text_model.embeddings.position_ids"] = torch.arange(77).unsqueeze(0)  # 77 is the max length of the text

    return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
    # 嫌になるくらい違うぞ！
    def convert_key(key):
        if not key.startswith("cond_stage_model"):
            return None

        # common conversion
        key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
        key = key.replace("cond_stage_model.model.", "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = None  # 使われない???
        elif ".logit_scale" in key:
            key = None  # 使われない???
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        # remove resblocks 23
        if ".resblocks.23." in key:
            continue
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks.23." in key:
            continue
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # rename or add position_ids
    ANOTHER_POSITION_IDS_KEY = "text_model.encoder.text_model.embeddings.position_ids"
    if ANOTHER_POSITION_IDS_KEY in new_sd:
        # waifu diffusion v1.4
        position_ids = new_sd[ANOTHER_POSITION_IDS_KEY]
        del new_sd[ANOTHER_POSITION_IDS_KEY]
    else:
        position_ids = torch.Tensor([list(range(max_length))]).to(torch.int64)

    new_sd["text_model.embeddings.position_ids"] = position_ids
    return new_sd


# endregion


# region Diffusers->StableDiffusion の変換コード
# convert_diffusers_to_original_stable_diffusion をコピーして修正している（ASL 2.0）


def conv_transformer_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
    unet_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
    ]

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        # loop over downblocks/upblocks

        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            if i > 0:
                # no attention layers in up_blocks.0
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    if v2:
        conv_transformer_to_linear(new_state_dict)

    return new_state_dict


def controlnet_conversion_map():
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("middle_block_out.0.weight", "controlnet_mid_block.weight"),
        ("middle_block_out.0.bias", "controlnet_mid_block.bias"),
    ]

    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        for j in range(2):
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    controlnet_cond_embedding_names = ["conv_in"] + [f"blocks.{i}" for i in range(6)] + ["conv_out"]
    for i, hf_prefix in enumerate(controlnet_cond_embedding_names):
        hf_prefix = f"controlnet_cond_embedding.{hf_prefix}."
        sd_prefix = f"input_hint_block.{i*2}."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    for i in range(12):
        hf_prefix = f"controlnet_down_blocks.{i}."
        sd_prefix = f"zero_convs.{i}.0."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer


def convert_controlnet_state_dict_to_sd(controlnet_state_dict):
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict.keys()}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[diffusers_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(diffusers_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(diffusers_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


def convert_controlnet_state_dict_to_diffusers(controlnet_state_dict):
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict.keys()}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[sd_name] = diffusers_name
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(sd_part, diffusers_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "resnets" in v:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(sd_part, diffusers_part)
            mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    vae_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3-i}.upsample."
            vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

    # this part accounts for mid blocks in both the encoder and the decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i+1}."
        vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

    if diffusers.__version__ < "0.17.0":
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "query."),
            ("k.", "key."),
            ("v.", "value."),
            ("proj_out.", "proj_attn."),
        ]
    else:
        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "to_q."),
            ("k.", "to_k."),
            ("v.", "to_v."),
            ("proj_out.", "to_out.0."),
        ]

    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                # logger.print(f"Reshaping {k} for SD format: shape {v.shape} -> {v.shape} x 1 x 1")
                new_state_dict[k] = reshape_weight_for_sd(v)

    return new_state_dict


def get_model_version_str_for_sd1_sd2(v2, v_parameterization):
    # only for reference
    version_str = "sd"
    if v2:
        version_str += "_v2"
    else:
        version_str += "_v1"
    if v_parameterization:
        version_str += "_v"
    return version_str


def convert_text_encoder_state_dict_to_sd_v2(checkpoint, make_dummy_weights=False):
    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    # 最後の層などを捏造するか
    if make_dummy_weights:
        logger.print("make dummy weights for resblock.23, text_projection and logit scale.")
        keys = list(new_sd.keys())
        for key in keys:
            if key.startswith("transformer.resblocks.22."):
                new_sd[key.replace(".22.", ".23.")] = new_sd[key].clone()  # copyしないとsafetensorsの保存で落ちる

        # Diffusersに含まれない重みを作っておく
        new_sd["text_projection"] = torch.ones((1024, 1024), dtype=new_sd[keys[0]].dtype, device=new_sd[keys[0]].device)
        new_sd["logit_scale"] = torch.tensor(1)

    return new_sd


VAE_PREFIX = "first_stage_model."


def load_vae(vae_id, dtype):
    logger.print(f"load VAE: {vae_id}")
    if os.path.isdir(vae_id) or not os.path.isfile(vae_id):
        # Diffusers local/remote
        try:
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder=None, torch_dtype=dtype)
        except EnvironmentError as e:
            logger.print(f"exception occurs in loading vae: {e}")
            logger.print("retry with subfolder='vae'")
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=dtype)
        return vae

    # local
    vae_config = create_vae_diffusers_config()

    if vae_id.endswith(".bin"):
        # SD 1.5 VAE on Huggingface
        converted_vae_checkpoint = torch.load(vae_id, map_location="cpu")
    else:
        # StableDiffusion
        vae_model = load_file(vae_id, "cpu") if is_safetensors(vae_id) else torch.load(vae_id, map_location="cpu")
        vae_sd = vae_model["state_dict"] if "state_dict" in vae_model else vae_model

        # vae only or full model
        full_model = False
        for vae_key in vae_sd:
            if vae_key.startswith(VAE_PREFIX):
                full_model = True
                break
        if not full_model:
            sd = {}
            for key, value in vae_sd.items():
                sd[VAE_PREFIX + key] = value
            vae_sd = sd
            del sd

        # Convert the VAE model.
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae


# endregion


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))

        # # make additional resos
        # if width >= height and width - divisible >= min_size:
        #   resos.add((width - divisible, height))
        #   resos.add((height, width - divisible))
        # if height >= width and height - divisible >= min_size:
        #   resos.add((width, height - divisible))
        #   resos.add((height - divisible, width))

        size += divisible

    resos = list(resos)
    resos.sort()
    return resos


if __name__ == "__main__":
    resos = make_bucket_resolutions((512, 768))
    logger.print(len(resos))
    logger.print(resos)
    aspect_ratios = [w / h for w, h in resos]
    logger.print(aspect_ratios)

    ars = set()
    for ar in aspect_ratios:
        if ar in ars:
            logger.print("error! duplicate ar:", ar)
        ars.add(ar)
