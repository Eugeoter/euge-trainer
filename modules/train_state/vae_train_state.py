import torch
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import save_file
from waifuset import logging
from .base_train_state import BaseTrainState
from ..utils import sd15_model_utils


class VAETrainState(BaseTrainState):
    save_best_model: bool = True
    eval_valid_size: int = 4

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.best_loss = float('inf')

    def save_vae_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}" + self.get_save_extension())
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        vae = self.unwrap_model(self.vae)
        state_dict = vae.state_dict()
        state_dict = sd15_model_utils.convert_vae_state_dict(state_dict)

        if self.save_as_format == 'torch':
            torch.save(state_dict, save_path)
        elif self.save_as_format == 'safetensors':
            save_file(state_dict, save_path)
        else:
            raise ValueError(f"Invalid save format: {self.save_as_format}")

        self.logger.print(f"VAE model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def do_eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False, on_train_start=False):
        mse_losses = []
        for model in self.training_models:
            model.eval()
        pbar = self.logger.tqdm(total=len(self.valid_dataloader), desc="evaluating")
        with torch.no_grad():
            mse_losses = []
            sample_dir = os.path.join(self.output_dir, self.output_subdir.samples, f"ep{self.epoch}_step{self.global_step}")
            sample_count = 0
            os.makedirs(sample_dir, exist_ok=True)
            for step, batch in enumerate(self.valid_dataloader):
                images = batch["images"].to(self.device, dtype=self.weight_dtype)
                vae = self.unwrap_model(self.vae)
                latents = vae.encode(images).latent_dist.sample()
                latents *= self.vae_scale_factor
                latents = 1 / self.vae_scale_factor * latents
                pred_images = self.vae.decode(latents).sample
                pred_images = pred_images.clamp(-1, 1)

                for i, (orig_img, pred_img) in enumerate(zip(batch["images"], pred_images)):
                    if self.eval_valid_size and sample_count >= self.eval_valid_size:
                        break
                    save_path = os.path.join(sample_dir, f"sample_{sample_count}.png")
                    # concat images
                    orig_img = ((orig_img / 2 + 0.5) * 255).numpy().astype(np.uint8).transpose(1, 2, 0)
                    pred_img = ((pred_img / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                    # self.logger.debug(f"orig_img: {orig_img.shape}, pred_img: {pred_img.shape}")
                    # self.logger.debug(f"orig_img: {orig_img.min()}, {orig_img.max()}, pred_img: {pred_img.min()}, {pred_img.max()}")
                    concat_img = np.concatenate([orig_img, pred_img], axis=1)
                    # self.logger.debug(f"concat_img.shape: {concat_img.shape}, dtype: {concat_img.dtype}, type: {type(concat_img)}")
                    # self.logger.debug(f"concat_img.min(): {concat_img.min()}, concat_img.max(): {concat_img.max()}")
                    # self.logger.debug(f"concat_img: {concat_img}")
                    concat_img = Image.fromarray(concat_img)
                    concat_img.save(save_path)
                    sample_count += 1

                loss = F.mse_loss(pred_images.float(), images.clamp(-1, 1).float(), reduction="mean")

                mse_losses.append(loss.detach().item())
                pbar.update(1)
        pbar.close()
        mean_loss = sum(mse_losses) / len(mse_losses)

        self.logger.info(f"validation at epoch {logging.yellow(self.epoch)}")
        self.logger.info(f"avg MSE loss {logging.green(mean_loss,  format_spec='.4f')}")
        self.logger.info(f"sampled images saved to: `{logging.yellow(sample_dir)}`")

        if self.save_best_model:
            if mean_loss < self.best_loss:
                self.best_mae_loss = mean_loss
                self.logger.info(f"best MAE loss {logging.green(mean_loss, format_spec='.4f')} achieved, saving model...")
                self.do_save()
