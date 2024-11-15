import torch
import random
import numpy as np
import os
import warnings
import torch.nn.functional as F
import re
from typing import List, Optional, Union
from waifuset import logging


logger = logging.get_logger("advanced")


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.print(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, prediction_type):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if prediction_type == "v_prediction":
        snr_weight = torch.div(min_snr_gamma, snr+1).float().to(loss.device)
    elif prediction_type == 'epsilon':
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    else:
        raise NotImplementedError(f"prediction_type: {prediction_type}")
    loss = loss * snr_weight
    return loss


def scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler):
    # ? scale v-pred loss back to noise-pred loss
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.print(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.print(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss


def apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=False):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


def apply_masked_loss(loss, batch):
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)  # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss


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


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
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
        logger.print("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
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


def get_unweighted_text_embeddings(
    tokenizer,
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
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

            if clip_skip is None or clip_skip == 1:
                text_embedding = text_encoder(text_input_chunk)[0]
            else:
                enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)

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
        if clip_skip is None or clip_skip == 1:
            text_embeddings = text_encoder(text_input)[0]
        else:
            enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    clip_skip=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

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
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return text_embeddings


# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4):
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale):
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def logit_normal(mu, sigma, shape, device):
    z = torch.randn(*shape, device=device)
    z = mu + sigma * z
    t = torch.sigmoid(z)
    return t


# Adjust Losses function is used to soften out difference in chances between highest and lowest loss timesteps.
# Otherwise difference can reach 10-20-50x between highest and lowest chances to sample.
def adjust_losses(all_losses, penalty=1.0):
    highest_loss = all_losses.max()
    lowest_loss = all_losses.min()
    average_loss = all_losses.mean()

    range_to_highest = highest_loss - average_loss

    adjusted_losses = torch.zeros_like(all_losses)

    for i, loss in enumerate(all_losses):
        if loss > average_loss:
            fraction = (loss - average_loss) / (range_to_highest + 1e-6)
            adjust_down = penalty * fraction
            adjusted_losses[i] = loss * (1 - adjust_down)
        else:
            fraction_below = (average_loss - loss) / (average_loss - lowest_loss + 1e-6)
            adjust_up = penalty * fraction_below
            adjusted_losses[i] = loss * (1 + adjust_up)

    adjusted_losses *= all_losses.sum() / adjusted_losses.sum()

    return adjusted_losses


def timestep_attention(run_number, loss_map, max_timesteps, b_size, device):
    # global adjusted_probabilities_print
    mean, std = 0.00, 1.00

    if loss_map:
        all_timesteps = torch.arange(1, max_timesteps, device=device)
        all_losses = torch.tensor([loss_map.get(t.item(), 1) for t in all_timesteps], device=device)

        # Adjust the losses based on the specified criteria
        adjusted_losses = adjust_losses(all_losses)

        # Calculate new probabilities with the adjusted losses
        adjusted_probabilities = adjusted_losses / adjusted_losses.sum()

        sampled_indices = torch.multinomial(adjusted_probabilities, b_size, replacement=True)
        skewed_timesteps = all_timesteps[sampled_indices]
    else:
        # Generate log-normal samples for timesteps as the fallback
        lognorm_samples = torch.distributions.LogNormal(mean, std).sample((b_size,)).to(device)
        normalized_samples = lognorm_samples / lognorm_samples.max()
        skewed_timesteps = (normalized_samples * (max_timesteps - 1)).long()

    # Log the adjusted probabilities
    # dir_name = f"H:\\TimestepAttention\\run{run_number}"
    # if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)
    # List existing files and find the next available file number
    # existing_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f)) and 'run_probabilities_' in f]
    # highest_num = 0
    # for file_name in existing_files:
    #    parts = file_name.replace('run_probabilities_', '').split('.')
    #    try:
    #        num = int(parts[0])
    #        if num > highest_num:
    #            highest_num = num
    #    except ValueError:
        # This handles the case where the file name does not end with a number
    #        continue

    # Determine the filename for the new log
    # new_file_num = highest_num + 1
    # file_out = os.path.join(dir_name, f"run_probabilities_{new_file_num}.txt")

    # adjusted_probabilities_print = adjusted_probabilities.cpu().tolist()
    # timesteps_probs_str = ', '.join(map(str, adjusted_probabilities_print))
    # with open(file_out, 'w') as file:
    #    file.write(timesteps_probs_str + '\n')

    return skewed_timesteps, skewed_timesteps


def update_loss_map_ema(current_loss_map, new_losses, timesteps, update_fraction=0.5):
    if not isinstance(new_losses, torch.Tensor):
        new_losses = torch.tensor(new_losses, dtype=torch.float32)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.tensor.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint*")
    timesteps = torch.tensor(timesteps, dtype=torch.long, device=new_losses.device)
    # Initialize all timesteps with a basic loss value of 0.1 if they are not already present
    for ts in range(1, 1000):  # Assuming timesteps from 1 to 999
        if ts not in current_loss_map:
            current_loss_map[ts] = 1

    aggregated_losses = new_losses.view(-1).tolist()

    # Define decay weights for adjusting adjacent timesteps
    decay_weights = [0.01, 0.03, 0.09, 0.25, 0.35, 0.50]

    for i, timestep_value in enumerate(timesteps.tolist()):
        loss_value = aggregated_losses[i] if i < len(aggregated_losses) else 0.0

        # Correctly handle the definition of current_average_loss
        # This ensures it's always defined before being used
        current_average_loss = current_loss_map.get(timestep_value, 1)

        # Directly update the timestep with the specified update fraction
        new_average_loss = current_average_loss + (loss_value - current_average_loss) * update_fraction
        current_loss_map[timestep_value] = new_average_loss

        # Apply decayed adjustments for adjacent timesteps
        for offset, weight in enumerate(decay_weights, start=1):
            for direction in [-1, 1]:
                adjacent_timestep = timestep_value + direction * offset
                if 1 <= adjacent_timestep < 1000:  # Ensure it's within the valid range
                    if adjacent_timestep in current_loss_map:
                        adjacent_current_loss = current_loss_map[adjacent_timestep]
                        # Calculate the decayed loss value based on distance and main loss value
                        decayed_loss_value = loss_value - (loss_value - current_average_loss) * weight
                        # Apply update fraction to move towards the decayed loss value
                        new_adjacent_loss = adjacent_current_loss + (decayed_loss_value - adjacent_current_loss) * update_fraction
                        current_loss_map[adjacent_timestep] = new_adjacent_loss

    return current_loss_map


def apply_loss_adjustment(loss, timesteps, loss_map, c_step, sched_train_steps):

    # Retrieve and calculate probabilities from the loss map
    all_timesteps = torch.arange(1, 1000, device=loss.device)  # Assuming max_timesteps is 1000
    all_losses = torch.tensor([loss_map.get(t.item(), 1.0) for t in all_timesteps], dtype=torch.float32, device=loss.device)

    # Calculate adjusted probabilities for each timestep
    adjusted_probabilities = all_losses / all_losses.sum()

    # Calculate the mean probability (average selection chance)
    mean_probability = adjusted_probabilities.mean()

    # Retrieve probabilities for specific sampled timesteps
    timestep_probabilities = adjusted_probabilities[timesteps - 1]

    # Calculate multipliers based on probabilities relative to the mean
    multipliers = timestep_probabilities / mean_probability

    schedule_start = 1
    schedule_move = -1
    loss_curve_scale = schedule_start + (schedule_move * (c_step/sched_train_steps))
    # Apply the 'loss_curve_scale' to modulate the multiplier effect
    # This reduces the effect of extreme multipliers, both high and low
    multipliers = 1 + (multipliers - 1) * loss_curve_scale

    # Adjust loss for each timestep based on its multiplier
    adjusted_loss = loss * multipliers.view(-1, 1, 1, 1)

    return adjusted_loss


def exponential_weighted_loss(noise_pred, target, alphas_cumprod, timesteps, loss_map, reduction="none", boundary_shift=0.0):
    # Compute the MAE loss
    mae_loss = F.l1_loss(noise_pred, target, reduction="none")

    # Calculate the mean of the MAE loss
    mean_mae_loss = mae_loss.mean()

    # Apply boundary shift if any
    mean_mae_loss += boundary_shift

    # Create a weighting map based on the exponential differences around the mean
    weight_map = torch.exp(mae_loss - mean_mae_loss)

    # Apply the weight map to the MAE loss
    weighted_loss = mae_loss * weight_map

    # Compute the mean and standard deviation along the spatial dimensions
    mean_weighted_loss = weighted_loss.mean(dim=(2, 3), keepdim=True)
    std_weighted_loss = weighted_loss.std(dim=(2, 3), keepdim=True)

    # Select the appropriate alphas_cumprod values based on timesteps
    ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

    # Compute the final weighted loss with alphas_cumprod adjustments
    final_weighted_loss = mean_weighted_loss * ac + std_weighted_loss * (1 - ac)

    # Compute the MSE loss
    mse_loss = F.mse_loss(noise_pred, target, reduction="none")

    # Compute the mean of the final weighted loss
    mean_final_weighted_loss = final_weighted_loss.mean()

    # Create masks for values below and above the mean
    below_mean_mask = final_weighted_loss < mean_final_weighted_loss
    above_mean_mask = ~below_mean_mask

    # Initialize the loss tensor
    loss = torch.zeros_like(final_weighted_loss)

    # Ensure mse_loss has compatible dimensions for masking
    mse_loss_mean = mse_loss.mean(dim=(2, 3), keepdim=True)

    # Replace values in the final weighted loss below the mean with even lower values of MSE
    loss[below_mean_mask] = torch.min(final_weighted_loss[below_mean_mask], mse_loss_mean.expand_as(final_weighted_loss)[below_mean_mask])

    # Interpolate values above the mean with their MSE counterpart by half
    loss[above_mean_mask] = 0.85 * final_weighted_loss[above_mean_mask] + 0.15 * mse_loss_mean.expand_as(final_weighted_loss)[above_mean_mask]

    # Normalize the loss map to create interpolation factors
    all_loss_values = torch.tensor(list(loss_map.values()), dtype=torch.float32, device=loss.device)
    min_loss_value = all_loss_values.min()
    max_loss_value = all_loss_values.max()

    # Map timesteps to normalized interpolation factors
    normalized_factors = torch.tensor([loss_map.get(t.item(), 1.0) for t in timesteps], dtype=torch.float32, device=loss.device)
    interpolation_factors = (max_loss_value - normalized_factors) / (max_loss_value - min_loss_value)

    # Apply median-based interpolation
    median_interpolation_factors = interpolation_factors.median()
    scaled_interpolation_factors = 0.5 * interpolation_factors + 0.5 * median_interpolation_factors

    # Further interpolate the loss using the scaled interpolation factors
    loss = (1 - scaled_interpolation_factors.view(-1, 1, 1, 1)) * mse_loss + scaled_interpolation_factors.view(-1, 1, 1, 1) * loss

    return loss


def adaptive_clustered_mse_loss(input, target, timesteps, loss_map, reduction="none", min_clusters=4, max_clusters=100):
    from sklearn.cluster import KMeans
    # Ensure input and target are tensors and on the same device
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError("Input and target must be tensors")

    if input.size() != target.size():
        raise ValueError("Input and target must have the same shape")

    device = input.device
    target = target.to(device)

    batch_size = input.size(0)
    adjusted_loss = torch.zeros_like(input, dtype=torch.float32)

    for i in range(batch_size):
        # Compute the initial element-wise squared difference for the i-th item in the batch
        initial_loss = (input[i] - target[i]) ** 2

        # Determine the number of clusters based on the loss map
        timestep_loss = loss_map.get(timesteps[i].item(), 1.0)
        n_clusters = min_clusters + (timestep_loss - min(loss_map.values())) / (max(loss_map.values()) - min(loss_map.values())) * (max_clusters - min_clusters)
        n_clusters = max(min(int(n_clusters), max_clusters), min_clusters)

        # Flatten the loss tensor to 1D and move to CPU for k-means
        loss_flat = initial_loss.view(-1).detach().cpu().numpy().reshape(-1, 1)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(loss_flat)

        # Reshape clusters back to the original shape
        clusters = clusters.reshape(initial_loss.shape)

        # Convert clusters to tensor and move to the same device as input
        clusters = torch.tensor(clusters, device=device, dtype=torch.long)

        # Compute mean loss for each cluster and adjust loss values
        unique_clusters = torch.unique(clusters)
        adjusted_loss_i = torch.zeros_like(initial_loss)
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster).float()
            cluster_loss = initial_loss * cluster_mask
            cluster_mean_loss = cluster_loss.sum() / cluster_mask.sum()  # Average loss for the cluster
            adjusted_loss_i += cluster_mask * cluster_mean_loss

        adjusted_loss[i] = adjusted_loss_i

    # Apply the reduction
    if reduction == 'mean':
        return adjusted_loss.mean()
    elif reduction == 'sum':
        return adjusted_loss.sum()
    elif reduction == 'none':
        return adjusted_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
