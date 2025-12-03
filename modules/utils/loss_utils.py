import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    def __init__(self, scales=[1.0, 0.5, 0.25], projection_dim=64):
        super().__init__()
        self.scales = scales
        self.projection_dim = projection_dim

    def sliced_wasserstein(self, z1, z2, num_projections=50):
        B, C, H, W = z1.shape
        device = z1.device

        # 通道维度投影
        proj_matrix = torch.randn(C, self.projection_dim, device=device)
        proj_matrix /= torch.norm(proj_matrix, dim=0, keepdim=True)

        z1_flat = torch.einsum('bchw,cd->bdhw', z1, proj_matrix).flatten(2)  # [B, D, H*W]
        z2_flat = torch.einsum('bchw,cd->bdhw', z2, proj_matrix).flatten(2)

        # 多方向投影排序对齐
        z1_sorted = torch.sort(z1_flat, dim=-1)[0]
        z2_sorted = torch.sort(z2_flat, dim=-1)[0]

        return F.mse_loss(z1_sorted, z2_sorted)

    def forward(self, pred, target):
        loss = 0.0
        for scale in self.scales:
            if scale != 1.0:
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='nearest')
                target_scaled = F.interpolate(target, scale_factor=scale, mode='nearest')
            else:
                pred_scaled = pred
                target_scaled = target

            loss += self.sliced_wasserstein(pred_scaled, target_scaled)
        return loss / len(self.scales)


class FrequencyLoss(nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight

    def forward(self, pred, target):
        assert pred.shape == target.shape, f"Pred shape {pred.shape} != Target shape {target.shape}"

        # Pad to even dimensions
        H, W = pred.shape[-2], pred.shape[-1]
        pad_H = (H % 2)
        pad_W = (W % 2)
        pred_padded = F.pad(pred, (0, pad_W, 0, pad_H))
        target_padded = F.pad(target, (0, pad_W, 0, pad_H))

        # Fast Fourier Transform
        pred_fft = torch.fft.rfft2(pred_padded, norm='ortho')
        target_fft = torch.fft.rfft2(target_padded, norm='ortho')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        _, _, H_freq, W_freq = pred_amp.shape
        y_freq = torch.linspace(0, 1, H_freq, device=pred.device)
        x_freq = torch.linspace(0, 1, W_freq, device=pred.device)
        freq_weights = 1 + (self.high_freq_weight-1) * (y_freq[:, None] + x_freq[None, :])

        return F.l1_loss(pred_amp * freq_weights, target_amp * freq_weights)
