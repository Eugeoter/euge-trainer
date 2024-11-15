import torch
import math
import os
import re
import gc
from accelerate import init_empty_weights
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from typing import Optional, List, Union
from waifuset import logging
from . import sd15_model_utils, train_utils, sdxl_model_utils
from ..models.sdxl import nnet as sdxl_original_unet

logger = logging.get_logger("train")

VAE_SCALE_FACTOR = 0.13025

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")

    train_utils.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_utils.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def get_block_params_to_optimize(unet, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def load_sdxl_tokenizers(tokenizer_cache_dir=None, max_token_length=75):
    name_or_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokenizers = []
    for i, name_or_path in enumerate(name_or_paths):
        tokenizer: CLIPTokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, name_or_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.print(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(name_or_path)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokenizers.append(tokenizer)

    if max_token_length is not None:
        logger.print(f"update token length: {max_token_length}")

    return tokenizers


def load_target_model(config, accelerator, model_version: str, weight_dtype):
    # load models for each process
    model_dtype = train_utils.match_mixed_precision(config, weight_dtype)  # prepare fp16/bf16
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}", disable=False)

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                config.pretrained_model_name_or_path,
                config.vae,
                model_version,
                weight_dtype,
                accelerator.device,
                model_dtype,
            )

            # work on low-ram device
            if config.cpu:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    name_or_path: str, vae_path: str, model_version: str, weight_dtype, device="cpu", model_dtype=None
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        logger.print(f"load StableDiffusion checkpoint: {name_or_path}")
        models = sdxl_model_utils.load_models_from_sdxl_checkpoint(name_or_path, device, model_dtype)
        unet, text_encoder1, text_encoder2, vae, logit_scale, ckpt_info = models['unet'], models['text_encoder1'], models['text_encoder2'], models['vae'], models['logit_scale'], models['ckpt_info']
    else:
        # Diffusers model is loaded to CPU
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.print(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None
                )
            except EnvironmentError as ex:
                if variant is not None:
                    logger.print("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            logger.print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        # convert to fp32 for cache text_encoders outputs
        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        state_dict = sdxl_model_utils.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel()  # overwrite unet
        sdxl_model_utils._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.print("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if vae_path is not None:
        vae = sd15_model_utils.load_vae(vae_path, weight_dtype)
        logger.print("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    pool2 = pool_workaround(text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i: i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i: i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def save_sdxl_model_during_train(
    config,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    save_path = os.path.join(config.output_dir, config.output_subdir.models, f"{os.path.basename(config.output_dir)}_ep{epoch}_step{global_step}.safetensors")
    sdxl_model_utils.save_stable_diffusion_checkpoint(
        save_path,
        text_encoder1,
        text_encoder2,
        unet,
        epoch,
        global_step,
        ckpt_info,
        vae,
        logit_scale,
        save_dtype,
    )
    return save_path


# def encode_caption(

#     captions,
#     negative_captions=None,
# ):
#     text_embeddings1, text_pool1, uncond_embeddings1, uncond_pool1 = encode_caption(
#         captions,
#         negative_captions=negative_captions,
#         is_sdxl_text_encoder2=False,
#     )
#     text_embeddings2, text_pool2, uncond_embeddings2, uncond_pool2 = encode_caption(
#         captions,
#         negative_captions=negative_captions,
#         is_sdxl_text_encoder2=True,
#     )
#     return text_embeddings1, text_embeddings2, text_pool2, uncond_embeddings1, uncond_embeddings2, uncond_pool2


def encode_caption(
    text_encoder,
    tokenizer,
    captions,
    negative_captions=None,
    max_embeddings_multiples=3,
    clip_skip=False,
    do_classifier_free_guidance=False,
    is_sdxl_text_encoder2=False,
    skip_parsing=True,
    device=None,
):
    device = device or text_encoder.device
    batch_size = len(captions) if isinstance(captions, list) else 1

    if negative_captions is None:
        negative_captions = [""] * batch_size
    elif isinstance(negative_captions, str):
        negative_captions = [negative_captions] * batch_size
    if batch_size != len(negative_captions):
        raise ValueError(
            f"`negative_captions`: {negative_captions} has batch size {len(negative_captions)}, but `prompt`:"
            f" {captions} has batch size {batch_size}. Please make sure that passed `negative_captions` matches"
            " the batch size of `prompt`."
        )

    text_embeddings, text_pool, uncond_embeddings, uncond_pool = get_weighted_text_embeddings(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        captions=captions,
        negative_captions=negative_captions if do_classifier_free_guidance else None,
        max_embeddings_multiples=max_embeddings_multiples,
        clip_skip=clip_skip,
        is_sdxl_text_encoder2=is_sdxl_text_encoder2,
        skip_parsing=True,
        device=device,
    )

    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.view(bs_embed, seq_len, -1)
    if text_pool is not None:
        text_pool = text_pool.view(bs_embed, -1)

    if do_classifier_free_guidance:
        bs_embed, seq_len, _ = uncond_embeddings.shape
        uncond_embeddings = uncond_embeddings.view(bs_embed, seq_len, -1)
        if uncond_pool is not None:
            uncond_pool = uncond_pool.view(bs_embed, -1)

        return text_embeddings, text_pool, uncond_embeddings, uncond_pool

    return text_embeddings, text_pool, None, None


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_captions_with_weights(tokenizer, captions: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in captions:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase `max_embeddings_multiples`.")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_hidden_states(text_encoder, input_ids, is_sdxl_text_encoder2: bool, eos_token_id, device):
    if not is_sdxl_text_encoder2:
        # text_encoder1: same as SD1/2
        enc_out = text_encoder(input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=True)
        hidden_states = enc_out["hidden_states"][11]
        pool = None
    else:
        # text_encoder2
        enc_out = text_encoder(input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=True)
        hidden_states = enc_out["hidden_states"][-2]  # penuultimate layer
        # pool = enc_out["text_embeds"]
        pool = pool_workaround(text_encoder, enc_out["last_hidden_state"], input_ids, eos_token_id)
    hidden_states = hidden_states.to(device)
    if pool is not None:
        pool = pool.to(device)
    return hidden_states, pool


def get_unweighted_text_embeddings(
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    is_sdxl_text_encoder2: bool,
    no_boseos_middle: Optional[bool] = True,
    device=None,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    device = device or text_encoder.device
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    text_pool = None
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            text_embedding, current_text_pool = get_hidden_states(
                text_encoder, text_input_chunk, is_sdxl_text_encoder2, eos, device
            )
            if text_pool is None:
                text_pool = current_text_pool

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings, text_pool = get_hidden_states(text_encoder, text_input, is_sdxl_text_encoder2, eos, device)
    return text_embeddings, text_pool


def get_weighted_text_embeddings(
    text_encoder,
    tokenizer,
    captions: Union[str, List[str]],
    negative_captions: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    is_sdxl_text_encoder2=False,
    device=None,
):
    device = device or text_encoder.device
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(captions, str):
        captions = [captions]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_captions_with_weights(tokenizer, captions, max_length - 2)
        if negative_captions is not None:
            if isinstance(negative_captions, str):
                negative_captions = [negative_captions]
            uncond_tokens, uncond_weights = get_captions_with_weights(tokenizer, negative_captions, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in tokenizer(captions, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if negative_captions is not None:
            if isinstance(negative_captions, str):
                negative_captions = [negative_captions]
            uncond_tokens = [
                token[1:-1] for token in tokenizer(negative_captions, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if negative_captions is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    if negative_captions is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings, text_pool = get_unweighted_text_embeddings(
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        is_sdxl_text_encoder2,
        no_boseos_middle=no_boseos_middle,
        device=device,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    if negative_captions is not None:
        uncond_embeddings, uncond_pool = get_unweighted_text_embeddings(
            text_encoder,
            uncond_tokens,
            tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            is_sdxl_text_encoder2,
            no_boseos_middle=no_boseos_middle,
            device=device,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if negative_captions is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if negative_captions is not None:
        return text_embeddings, text_pool, uncond_embeddings, uncond_pool
    return text_embeddings, text_pool, None, None
