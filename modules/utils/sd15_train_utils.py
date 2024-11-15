import torch

VAE_SCALE_FACTOR = 0.18215


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2  # = alpha^2 / (1 - alpha^2)

    noise_scheduler.all_snr = all_snr.to(device)


def get_input_ids(caption, tokenizer, max_token_length):
    input_ids = tokenizer(
        caption, padding="max_length", truncation=True, max_length=max_token_length, return_tensors="pt"
    ).input_ids

    if max_token_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        ids_list = []
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # v1
            # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
            # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
            for i in range(0, max_token_length + 2 - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                ids_list.append(ids_chunk)
        else:
            # v2 or SDXL
            # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
            for i in range(0, max_token_length + 2 - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = (
                    input_ids[0].unsqueeze(0),  # BOS
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )  # PAD or EOS
                ids_chunk = torch.cat(ids_chunk)

                # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                    ids_chunk[-1] = tokenizer.eos_token_id
                # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id

                ids_list.append(ids_chunk)

        input_ids = torch.stack(ids_list)  # 3,77
        return input_ids


def get_hidden_states(input_ids, tokenizer, text_encoder, weight_dtype=None, v2=False, clip_skip=None, max_token_length=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length is not None:
        if v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                states_list.append(encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def apply_weighted_noise(noise, mask, weight, normalize=True):
    # noise is [H, W, C] and mask is [H, W]
    mask = torch.stack([mask] * noise.shape[1], dim=1)
    noise = torch.where(mask > 0, noise * weight, noise)
    if normalize:
        noise = noise / noise.std()
    return noise
