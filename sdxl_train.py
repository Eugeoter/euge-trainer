import torch
import os
import math
import gc
import time
import traceback
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
from modules import advanced_train_utils, sdxl_train_utils, arg_utils, log_utils as logu


def train(args):
    accelerator = sdxl_train_utils.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process
    local_process_index = accelerator.state.local_process_index
    num_processes = accelerator.state.num_processes

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = sdxl_train_utils.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_utils.load_target_model(args, accelerator, "sdxl", weight_dtype)

    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(args.tokenizer_cache_dir, args.max_token_length)

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    dataset = sdxl_train_utils.load_dataset(
        args,
        tokenizer1,
        tokenizer2,
        latents_dtype=weight_dtype,
        is_main_process=is_main_process,
        num_processes=num_processes,
        process_idx=local_process_index
    )

    if args.cache_latents:
        with torch.no_grad():
            dataset.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, check_validity=args.check_cache_validity, async_cache=args.async_cache)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()

    dataloader_n_workers = min(args.max_dataloader_n_workers, os.cpu_count() - 1)
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=dataloader_n_workers,
        shuffle=True,
        collate_fn=sdxl_train_utils.collate_fn,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.diffusers_xformers:
        sdxl_train_utils.set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        sdxl_train_utils.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    training_models = []
    params_to_optimize = []

    train_unet = args.learning_rate > 0

    if args.block_lr is not None:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == sdxl_train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {sdxl_train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR} values"
    else:
        block_lrs = None

    if train_unet:
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        unet.requires_grad_(True)
        training_models.append(unet)
        if args.block_lr is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(sdxl_train_utils.get_block_params_to_optimize(unet, block_lrs))
    else:
        unet.requires_grad_(False)
        # because of unet is not prepared
        unet.to(accelerator.device, dtype=weight_dtype)

    if args.train_text_encoder:
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 or args.learning_rate
        lr_te2 = args.learning_rate_te2 or args.learning_rate
        train_text_encoder1 = lr_te1 > 0
        train_text_encoder2 = lr_te2 > 0
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
            training_models.append(text_encoder1)
            params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": lr_te1})
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
            training_models.append(text_encoder2)
            params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": lr_te2})
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        train_text_encoder1 = False
        train_text_encoder2 = False
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

    num_train_epochs = args.num_train_epochs
    num_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps / num_processes)
    num_train_steps = num_train_epochs * num_steps_per_epoch

    # Ensure weight dtype when full fp16/bf16 training
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    if train_unet:
        unet = accelerator.prepare(unet)
        (unet,) = sdxl_train_utils.transform_models_if_DDP([unet])
    if train_text_encoder1:
        text_encoder1 = accelerator.prepare(text_encoder1)
        (text_encoder1,) = sdxl_train_utils.transform_models_if_DDP([text_encoder1])
        text_encoder1.to(accelerator.device)
    if train_text_encoder2:
        text_encoder2 = accelerator.prepare(text_encoder2)
        (text_encoder2,) = sdxl_train_utils.transform_models_if_DDP([text_encoder2])
        text_encoder2.to(accelerator.device)

    if args.full_fp16:
        sdxl_train_utils.patch_accelerator_for_fp16_training(accelerator)

    _, _, optimizer = sdxl_train_utils.get_optimizer(args, params_to_optimize)

    lr_scheduler = sdxl_train_utils.get_scheduler_fix(args, optimizer, num_train_steps)

    optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        optimizer, lr_scheduler, train_dataloader
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    sdxl_train_utils.prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        advanced_train_utils.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if is_main_process and args.logging_dir is not None:
        accelerator.init_trackers("finetuning", init_kwargs={})

    progress_bar = tqdm(total=num_train_steps, smoothing=0, desc='steps', disable=not accelerator.is_local_main_process)
    global_step = 0
    loss_recorder = sdxl_train_utils.LossRecorder(length=args.loss_record_stride)

    accelerator.print(f"starting learning")
    accelerator.print(f"  device: {accelerator.device}")
    accelerator.print(f"  learning rate: {args.learning_rate} | te1: {args.learning_rate_te1 if train_text_encoder1 else 0} | te2: {args.learning_rate_te2 if train_text_encoder2 else 0}")
    accelerator.print(f"  batch size: {args.batch_size} | optimizer: {args.optimizer_type.lower()}")
    accelerator.print(f"  mixed precision: {args.mixed_precision} | weight-dtype: {weight_dtype} | save-dtype: {save_dtype}")
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  num train steps: {num_train_steps}")
    accelerator.print(f"  num train epochs: {num_train_epochs}")
    accelerator.print(f"  num steps per epoch: {num_steps_per_epoch}")

    try:
        for epoch in range(num_train_epochs):
            if is_main_process:
                progress_bar.write(f"epoch: {epoch + 1}/{num_train_epochs}")
            for m in training_models:
                m.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(*training_models):
                    if batch.get("latents") is not None:
                        latents = batch["latents"].to(accelerator.device)
                        if latents.dtype != weight_dtype:
                            latents = latents.to(weight_dtype)
                    else:
                        # print(f"missing latents in batch {batch['image_keys']}")
                        with torch.no_grad():
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)
                            if torch.any(torch.isnan(latents)):
                                progress_bar.write("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    latents *= sdxl_train_utils.VAE_SCALE_FACTOR

                    if batch.get("text_encoder_outputs1_list") is None:  # TODO: Implement text encoder cache
                        input_ids1 = batch["input_ids_1"]
                        input_ids2 = batch["input_ids_2"]
                        with torch.set_grad_enabled(args.train_text_encoder):
                            input_ids1 = input_ids1.to(accelerator.device)
                            input_ids2 = input_ids2.to(accelerator.device)
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_utils.get_hidden_states_sdxl(
                                args.max_token_length,
                                input_ids1,
                                input_ids2,
                                tokenizer1,
                                tokenizer2,
                                text_encoder1,
                                text_encoder2,
                                None if not args.full_fp16 else weight_dtype,
                            )
                    else:
                        encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                        encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                        pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                    target_size = batch["target_size_hw"]
                    orig_size = batch["original_size_hw"]
                    crop_size = batch["crop_top_lefts"]
                    embs = sdxl_train_utils.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                    vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                    noise, noisy_latents, timesteps = sdxl_train_utils.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                    noisy_latents = noisy_latents.to(weight_dtype)

                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                    # Deprecated
                    # if args.weighted_noise_type == 'canny':
                    #     canny = batch["canny"].to(accelerator.device).to(dtype=weight_dtype)
                    #     noise = sdxl_train_utils.apply_weighted_noise(noise, canny, weight=args.weighted_noise_weight)

                    target = noise

                    if (
                        args.min_snr_gamma
                        or args.scale_v_pred_loss_like_noise_pred
                        or args.v_pred_like_loss
                        or args.debiased_estimation_loss
                    ):
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])

                        if args.min_snr_gamma:
                            loss = advanced_train_utils.apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = advanced_train_utils.scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.v_pred_like_loss:
                            loss = advanced_train_utils.add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                        if args.debiased_estimation_loss:
                            loss = advanced_train_utils.apply_debiased_estimation(loss, timesteps, noise_scheduler)

                        loss = loss.mean()  # mean over batch dimension
                    else:
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    if torch.isnan(loss):
                        progress_bar.write("NaN found in loss, replacing with zeros")
                        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    sdxl_train_utils.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        [tokenizer1, tokenizer2],
                        [text_encoder1, text_encoder2],
                        unet,
                    )

                # loggings
                current_loss = loss.detach().item()
                if args.logging_dir is not None:
                    logs = {"loss": current_loss}
                    if block_lrs is None:
                        sdxl_train_utils.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                    else:
                        sdxl_train_utils.append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs
                    accelerator.log(logs, step=global_step)
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                pbar_logs = {
                    'lr': lr_scheduler.get_last_lr()[0],
                    'epoch': epoch + 1,
                    'next': num_steps_per_epoch - step - 1,
                    'avr_loss': avr_loss,
                    'cur_loss': current_loss,
                    # 'batch': batch['image_keys'][0],
                }
                progress_bar.set_postfix(pbar_logs)

                # del batch, latents, orig_size, target_size, crop_size, input_ids1, input_ids2, encoder_hidden_states1, encoder_hidden_states2, pool2, embs, vector_embedding, text_embedding, noise, noisy_latents, timesteps, noise_pred, target, loss

                if is_main_process and args.save_every_n_steps and (global_step + 1) % args.save_every_n_steps == 0:
                    progress_bar.write(f"saving model at step {global_step + 1}")
                    p = sdxl_train_utils.save_sd_model(
                        args,
                        save_dtype,
                        epoch + 1,
                        global_step + 1,
                        accelerator.unwrap_model(text_encoder1),
                        accelerator.unwrap_model(text_encoder2),
                        accelerator.unwrap_model(unet),
                        vae,
                        logit_scale,
                        ckpt_info
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    progress_bar.write(f"model saved at `{p}`")

            # end of epoch

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()
            if args.save_every_n_epochs and is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
                if is_main_process:
                    progress_bar.write(f"saving model at epoch {epoch + 1}")
                    p = sdxl_train_utils.save_sd_model(
                        args,
                        save_dtype,
                        epoch + 1,
                        global_step + 1,
                        accelerator.unwrap_model(text_encoder1),
                        accelerator.unwrap_model(text_encoder2),
                        accelerator.unwrap_model(unet),
                        vae,
                        logit_scale,
                        ckpt_info
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    progress_bar.write(f"model saved at `{p}`")

            sdxl_train_utils.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                [tokenizer1, tokenizer2],
                [text_encoder1, text_encoder2],
                unet,
            )

    except KeyboardInterrupt:
        do_save = is_main_process and args.save_on_keyboard_interrupt
        print("KeyboardInterrupted.")
    except Exception as e:
        do_save = is_main_process and args.save_on_exception
        print("Exception:", e)
        traceback.print_exc()
    else:
        do_save = is_main_process and args.save_on_train_end

    progress_bar.close()

    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if do_save and is_main_process:
        logu.info("saving model on train end...")
        p = sdxl_train_utils.save_sd_model(
            args,
            save_dtype,
            epoch + 1,
            global_step + 1,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        logu.success(f"model saved at {p}")

    logu.success(f"training finished at process {local_process_index+1}/{num_processes}")
    del accelerator


if __name__ == "__main__":
    args = arg_utils.parse_arguments()
    train(args)
