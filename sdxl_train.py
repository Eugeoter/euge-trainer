import torch
import torch.distributed
import os
import math
import gc
import traceback
from absl import flags
from absl import app
from ml_collections import config_flags
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
from modules import advanced_train_utils, sdxl_train_utils, sdxl_eval_utils, sdxl_dataset_utils, log_utils
from modules.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline


def train(argv):
    config = flags.FLAGS.config
    config.output_dir = log_utils.smart_path(os.path.dirname(config.output_dir), os.path.basename(config.output_dir))
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    accelerator = sdxl_train_utils.prepare_accelerator(config)

    is_main_process = accelerator.is_main_process
    local_process_index = accelerator.state.local_process_index
    num_processes = accelerator.state.num_processes

    logger = log_utils.get_logger("train", disable=not is_main_process)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = sdxl_train_utils.prepare_dtype(config)
    vae_dtype = torch.float32 if config.no_half_vae else weight_dtype

    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_utils.load_target_model(config, accelerator, "sdxl", weight_dtype)

    logger.print("prepare tokenizers")
    tokenizer1, tokenizer2 = sdxl_train_utils.load_tokenizers(config.tokenizer_cache_dir, config.max_token_length)

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    logger.print(f"prepare dataset...")
    dataset = sdxl_dataset_utils.Dataset(
        config=config,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        latents_dtype=weight_dtype,
        is_main_process=is_main_process,
        num_processes=num_processes,
        process_idx=local_process_index,
    )

    if config.cache_latents:
        with torch.no_grad():
            dataset.cache_latents(vae, accelerator, config.vae_batch_size, config.cache_latents_to_disk, check_validity=config.check_cache_validity, async_cache=config.async_cache)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()

    dataloader_n_workers = min(config.max_dataloader_n_workers, os.cpu_count() - 1)
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,  # fix to 1 because collate_fn returns a dict
        num_workers=dataloader_n_workers,
        shuffle=True,
        collate_fn=sdxl_train_utils.collate_fn,
        persistent_workers=config.persistent_data_loader_workers,
    )

    if config.diffusers_xformers:
        sdxl_train_utils.set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        logger.print("Disable Diffusers' xformers")
        sdxl_train_utils.replace_unet_modules(unet, config.mem_eff_attn, config.xformers, config.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(config.xformers)

    training_models = []
    params_to_optimize = []

    train_unet = config.learning_rate > 0

    if config.block_lr is not None:
        block_lrs = [float(lr) for lr in config.block_lr.split(",")]
        assert (
            len(block_lrs) == sdxl_train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {sdxl_train_utils.UNET_NUM_BLOCKS_FOR_BLOCK_LR} values"
    else:
        block_lrs = None

    if train_unet:
        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        unet.requires_grad_(True)
        training_models.append(unet)
        if config.block_lr is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": config.learning_rate})
        else:
            params_to_optimize.extend(sdxl_train_utils.get_block_params_to_optimize(unet, block_lrs))
    else:
        unet.requires_grad_(False)
        # because of unet is not prepared
        unet.to(accelerator.device, dtype=weight_dtype)

    if config.train_text_encoder:
        if config.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = config.learning_rate_te1 or config.learning_rate
        lr_te2 = config.learning_rate_te2 or config.learning_rate
        train_text_encoder1 = lr_te1 > 0
        train_text_encoder2 = lr_te2 > 0
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        else:
            training_models.append(text_encoder1)
            params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": lr_te1})
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        else:
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

    total_batch_size = config.batch_size * config.gradient_accumulation_steps * num_processes
    num_train_epochs = config.num_train_epochs
    num_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps / num_processes)
    num_train_steps = num_train_epochs * num_steps_per_epoch

    # Ensure weight dtype when full fp16/bf16 training
    if config.full_fp16:
        assert (
            config.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        logger.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif config.full_bf16:
        assert (
            config.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        logger.print("enable full bf16 training.")
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

    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    if config.full_fp16:
        sdxl_train_utils.patch_accelerator_for_fp16_training(accelerator)

    _, _, optimizer = sdxl_train_utils.get_optimizer(config, params_to_optimize)

    lr_scheduler = sdxl_train_utils.get_scheduler_fix(config, optimizer, num_train_steps)

    optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        optimizer, lr_scheduler, train_dataloader
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    sdxl_train_utils.prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if config.zero_terminal_snr:
        advanced_train_utils.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if is_main_process:
        accelerator.init_trackers("finetuning", init_kwargs={})
    loss_recorder = sdxl_train_utils.LossRecorder(gamma=config.loss_recorder_kwargs.gamma)

    logger.print(log_utils.green(f"==================== START TRAINING ===================="))
    logger.print(f"  num train steps: {log_utils.yellow(num_train_epochs)} x {log_utils.yellow(num_steps_per_epoch)} = {log_utils.yellow(num_train_steps)}")
    logger.print(f"  train unet: {train_unet} | learning rate: {config.learning_rate}")
    logger.print(f"  train text encoder 1: {train_text_encoder1} | learning rate: {config.learning_rate_te1 if train_text_encoder1 else 0}")
    logger.print(f"  train text encoder 2: {train_text_encoder2} | learning rate: {config.learning_rate_te2 if train_text_encoder2 else 0}")
    logger.print(f"  number of trainable parameters: {n_params} = {n_params / 1e9:.1f}B")
    logger.print(
        f"  total batch size: {log_utils.yellow(total_batch_size)} = {config.batch_size} (batch size) x {config.gradient_accumulation_steps} (gradient accumulation steps) x {num_processes} (num processes)")
    logger.print(f"  mixed precision: {config.mixed_precision} | weight-dtype: {weight_dtype} | save-dtype: {save_dtype}")
    logger.print(f"  optimizer: {config.optimizer_type} | timestep sampler: {config.timestep_sampler_type}")
    logger.print(f"  device: {log_utils.yellow(accelerator.device)}")

    progress_bar = tqdm(total=num_train_steps, desc='steps', disable=not accelerator.is_local_main_process)
    global_step = 0
    # nan_cnt = 0

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
                        # if latents.dtype != weight_dtype:
                        #     latents = latents.to(weight_dtype)
                    else:
                        # logger.print(f"missing latents in batch {batch['image_keys']}")
                        with torch.no_grad():
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)
                            if torch.any(torch.isnan(latents)):
                                progress_bar.write("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    latents *= sdxl_train_utils.VAE_SCALE_FACTOR

                    if batch.get("text_encoder_outputs1_list") is None:  # TODO: Implement text encoder cache
                        input_ids1 = batch["input_ids_1"]
                        input_ids2 = batch["input_ids_2"]
                        with torch.set_grad_enabled(config.train_text_encoder):
                            input_ids1 = input_ids1.to(accelerator.device)
                            input_ids2 = input_ids2.to(accelerator.device)
                            encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_utils.get_hidden_states_sdxl(
                                config.max_token_length,
                                input_ids1,
                                input_ids2,
                                tokenizer1,
                                tokenizer2,
                                text_encoder1,
                                text_encoder2,
                                None if not config.full_fp16 else weight_dtype,
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

                    noise, noisy_latents, timesteps = sdxl_train_utils.get_noise_noisy_latents_and_timesteps(config, noise_scheduler, latents)

                    noisy_latents = noisy_latents.to(weight_dtype)

                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                        # if guess_mode and do_classifier_free_guidance:
                        #     # Infered ControlNet only for the conditional batch.
                        #     # To apply the output of ControlNet to both the unconditional and conditional batches,
                        #     # add 0 to the unconditional batch to keep it unchanged.
                        #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    target = noise

                    if (
                        config.min_snr_gamma
                        or config.scale_v_pred_loss_like_noise_pred
                        or config.v_pred_like_loss
                        or config.debiased_estimation_loss
                    ):
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])

                        if config.min_snr_gamma:
                            loss = advanced_train_utils.apply_snr_weight(loss, timesteps, noise_scheduler, config.min_snr_gamma)
                        if config.scale_v_pred_loss_like_noise_pred:
                            loss = advanced_train_utils.scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if config.v_pred_like_loss:
                            loss = advanced_train_utils.add_v_prediction_like_loss(loss, timesteps, noise_scheduler, config.v_pred_like_loss)
                        if config.debiased_estimation_loss:
                            loss = advanced_train_utils.apply_debiased_estimation(loss, timesteps, noise_scheduler)

                        loss = loss.mean()  # mean over batch dimension
                    else:
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    if torch.isnan(loss):
                        # progress_bar.write("NaN found in loss, replacing with zeros")
                        # nan_cnt += 1
                        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and config.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if config.sample_every_n_steps is not None and (global_step + 1) % config.sample_every_n_steps == 0:
                        try:
                            sdxl_eval_utils.sample_during_train(
                                pipe_class=SdxlStableDiffusionLongPromptWeightingPipeline,
                                accelerator=accelerator,
                                config=config,
                                epoch=None,
                                steps=global_step + 1,
                                unet=unet,
                                text_encoder=[text_encoder1, text_encoder2],
                                vae=vae,
                                tokenizer=[tokenizer1, tokenizer2],
                                device=accelerator.device,
                            )

                        except Exception as e:
                            logger.print("exception when sample images:", e)
                            traceback.print_exc()
                            pass

                    if config.save_every_n_steps is not None and (global_step + 1) % config.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if is_main_process:
                            progress_bar.write(f"saving model at step {global_step + 1}")
                            p = sdxl_train_utils.save_sdxl_model_during_train(
                                config,
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
                        accelerator.wait_for_everyone()

                # loggings
                step_loss: float = loss.detach().item()
                loss_recorder.add(loss=step_loss)
                avr_loss: float = loss_recorder.moving_average(window=config.loss_recorder_kwargs.stride)
                ema_loss: float = loss_recorder.ema

                logs = {"loss/step": step_loss, 'loss_avr/step': avr_loss, 'loss_ema/step': ema_loss}
                if block_lrs is None:
                    sdxl_train_utils.append_lr_to_logs(logs, lr_scheduler, config.optimizer_type, including_unet=train_unet)
                else:
                    sdxl_train_utils.append_block_lr_to_logs(block_lrs, logs, lr_scheduler, config.optimizer_type)  # U-Net is included in block_lrs
                accelerator.log(logs, step=global_step)

                pbar_logs = {
                    'lr': lr_scheduler.get_last_lr()[0],
                    'epoch': epoch + 1,
                    'global_step': global_step + 1,
                    'next': len(train_dataloader) - step - 1,
                    'step_loss': step_loss,
                    'avr_loss': avr_loss,
                    'ema_loss': ema_loss,
                    # 'nan': nan_cnt,
                    # 'batch': batch['image_keys'][0],
                }
                progress_bar.set_postfix(pbar_logs)

                # del batch, latents, orig_size, target_size, crop_size, input_ids1, input_ids2, encoder_hidden_states1, encoder_hidden_states2, pool2, embs, vector_embedding, text_embedding, noise, noisy_latents, timesteps, noise_pred, target, loss

            # end of epoch
            logs = {
                "loss/epoch": loss_recorder.moving_average(window=num_steps_per_epoch)
            }
            accelerator.log(logs, step=epoch + 1)
            accelerator.wait_for_everyone()

            if config.save_every_n_epochs and (epoch + 1) % config.save_every_n_epochs == 0:
                if is_main_process:
                    progress_bar.write(f"saving model at epoch {epoch + 1}")
                    p = sdxl_train_utils.save_sdxl_model_during_train(
                        config,
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
                # accelerator.wait_for_everyone()

            if config.sample_every_n_epochs and (epoch + 1) % config.sample_every_n_epochs == 0:
                try:
                    sdxl_eval_utils.sample_during_train(
                        pipe_class=SdxlStableDiffusionLongPromptWeightingPipeline,
                        accelerator=accelerator,
                        config=config,
                        epoch=epoch + 1,
                        steps=global_step,
                        vae=vae,
                        tokenizer=[tokenizer1, tokenizer2],
                        text_encoder=[text_encoder1, text_encoder2],
                        unet=unet,
                        device=accelerator.device,
                    )
                except Exception as e:
                    logger.print("exception when sample images:", e)
                    traceback.print_exc()
                    pass

    except KeyboardInterrupt:
        do_save = is_main_process and config.save_on_keyboard_interrupt
        logger.print("KeyboardInterrupted.")
    except Exception as e:
        do_save = is_main_process and config.save_on_exception
        logger.print("Exception:", e)
        traceback.print_exc()
    else:
        do_save = is_main_process and config.save_on_train_end

    progress_bar.close()

    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if do_save and is_main_process:
        logger.print("saving model on train end...")
        p = sdxl_train_utils.save_sdxl_model_during_train(
            config,
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
        logger.print(log_utils.green(f"model saved at {p}"))

    logger.print(log_utils.green(f"training finished at process {local_process_index+1}/{num_processes}"), disable=False)
    del accelerator


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    app.run(train)
