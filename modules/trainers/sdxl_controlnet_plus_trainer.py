import torch
from diffusers.utils.pil_utils import numpy_to_pil
from .sdxl_controlnet_trainer import SDXLControlNetTrainer


class SDXLControlNeXtPlusTrainer(SDXLControlNetTrainer):
    lambda_reward = {
        'segmentation': 0.5,
        'depth_lerest': 0.5,
        'depth_leres++': 0.5,
        'depth_midas': 0.5,
        'depth_zoe': 0.5,
        'normal_midas': 0.5,
        'normal_bae': 0.5,
        'hed': 1.0,
        'canny': 1.0,
        'lineart_anime': 10.0,
        'lineart_coarse': 10.0,
        'lineart_realistic': 10.0,
        'mlsd': 10.0,
        'softedge_hed': 1.0,
        'softedge_hedsafe': 1.0,
        'softedge_pidinet': 1.0,
        'softedge_pidsafe': 1.0,
        'scribble_hed': 1.0,
        'scribble_pidinet': 1.0,

    }
    timestep_threshold = 200

    def get_loss(self, model_pred, target, timesteps, batch) -> float:
        loss_train = super().get_loss(model_pred, target, timesteps, batch)
        # loss_reward = self.get_reward_loss(model_pred, target, timesteps, batch)
        loss_reward = 0.0
        return loss_train + loss_reward

    def get_reward_loss(self, model_pred, target, timesteps, batch) -> float:
        latents_approx = (target - torch.sqrt(1-self.noise_scheduler.alphas_cumprod[timesteps])[:, None, None, None] *
                          model_pred) / torch.sqrt(self.noise_scheduler.alphas_cumprod[timesteps])[:, None, None, None]

        images_approx = self.vae.decode(latents_approx.to(self.vae_dtype) / self.vae_scale_factor).sample
        images_approx = (images_approx / 2 + 0.5).clamp(0, 1)
        images_approx = images_approx.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        images_approx = numpy_to_pil(images_approx)
        condition_image_types = batch['condition_image_types']
        img_mds_approx = [
            {
                'image_key': None,
                'image': img,
                'condition_image_type': condition_image_type,
            }
            for img, condition_image_type in zip(images_approx, condition_image_types)
        ]
        condition_images_approx = [self.train_dataset.get_condition_image(img_md).to(self.device, dtype=torch.float32) for img_md in img_mds_approx]
        condition_images_approx = torch.stack(condition_images_approx, dim=0).to(memory_format=torch.contiguous_format).float()
        control_images = batch['condition_images'].to(self.device, dtype=torch.float32)
        reward_loss = 0.0
        for i in range(len(images_approx)):
            if timesteps[i] > self.timestep_threshold:
                continue
            condition_image_type = condition_image_types[i]
            lambda_r = self.lambda_reward.get(condition_image_type, 0.0)
            if lambda_r == 0.0:
                continue
            reward_loss += torch.nn.functional.mse_loss(
                condition_images_approx[i:i+1],
                control_images[i:i+1],
            ) * lambda_r

        # Release memory
        del condition_images_approx
        torch.cuda.empty_cache()

        return reward_loss / len(images_approx)
