import os
import torch
from waifuset import logging
from safetensors.torch import save_file
from .base_train_state import BaseTrainState
from ..utils import ws_train_utils


class WaifuScorerTrainState(BaseTrainState):
    save_best_model: bool = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.best_mse_loss = float('inf')
        self.best_mae_loss = float('inf')

    def save_mlp_model(self):
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_ep{self.epoch}_step{self.global_step}" + self.get_save_extension())
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.save_as_format == 'torch':
            torch.save(self.mlp.state_dict(), save_path)
        elif self.save_as_format == 'safetensors':
            save_file(self.mlp.state_dict(), save_path)
        else:
            raise ValueError(f"Invalid save format: {self.save_as_format}")

        self.logger.print(f"MLP model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def do_eval(self, on_step_end=False, on_epoch_end=False, on_train_end=False, on_train_start=False):
        for model in self.training_models:
            model.eval()
        with torch.no_grad():
            mse_losses = []
            mae_losses = []
            for step, batch in enumerate(self.valid_dataloader):
                # optimizer.zero_grad(set_to_none=True)
                if batch.get("image_embeddings") is not None:
                    img_embs = batch["image_embeddings"].to(self.device)
                else:
                    images = batch['images']
                    with torch.no_grad():
                        img_embs = ws_train_utils.encode_images(
                            images,
                            self.clip_model,
                            self.clip_preprocess,
                            device=self.device,
                            max_workers=self.train_dataset.max_dataset_n_workers
                        )

                with self.accelerator.autocast():
                    output = self.mlp(img_embs)

                # mae
                scores = batch['scores'].to(self.device)
                mse_loss = torch.mean((output - scores)**2)
                mae_loss = torch.mean(torch.abs(output - scores))
                mse_losses.append(mse_loss.detach().item())
                mae_losses.append(mae_loss.detach().item())
            mse_mean_loss = sum(mse_losses)/len(mse_losses)
            mae_mean_loss = sum(mae_losses)/len(mae_losses)
            self.logger.info(f"validation at epoch {logging.yellow(self.epoch)}")
            self.logger.info(f"avg MSE loss {logging.green(mse_mean_loss,  format_spec='.4f')}", no_prefix=True)
            self.logger.info(f"avg MAE loss {logging.yellow(mae_mean_loss, format_spec='.4f')}", no_prefix=True)

            if self.save_best_model:
                if mae_mean_loss < self.best_mae_loss:
                    self.best_mae_loss = mae_mean_loss
                    self.logger.info(f"best MAE loss {logging.green(mae_mean_loss, format_spec='.4f')} achieved, saving model...")
                    self.do_save()
