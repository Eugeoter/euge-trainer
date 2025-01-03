import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from torchvision import models

# Function to split the image into patches


def extract_patches(image, patch_size, stride):
    # Unfold the image into patches
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # Reshape to get a batch of patches
    patches = patches.contiguous().view(image.size(0), image.size(1), -1, patch_size, patch_size)
    return patches

# Patch-Based MSE Loss


def patch_based_mse_loss(real_images, recon_images, patch_size=32, stride=16):
    real_patches = extract_patches(real_images, patch_size, stride)
    recon_patches = extract_patches(recon_images, patch_size, stride)
    mse_loss = F.mse_loss(real_patches, recon_patches)
    return mse_loss

# Patch-Based LPIPS Loss (using the pre-defined LPIPS model)


def patch_based_lpips_loss(lpips_model, real_images, recon_images, patch_size=32, stride=16):
    with torch.no_grad():
        real_patches = extract_patches(real_images, patch_size, stride)
        recon_patches = extract_patches(recon_images, patch_size, stride)

        lpips_loss = 0
        # Iterate over each patch and accumulate LPIPS loss
        for i in range(real_patches.size(2)):  # Loop over number of patches
            real_patch = real_patches[:, :, i, :, :].contiguous()
            recon_patch = recon_patches[:, :, i, :, :].contiguous()
            patch_lpips_loss = lpips_model(real_patch, recon_patch).mean()

            # Handle non-finite values
            if not torch.isfinite(patch_lpips_loss):
                patch_lpips_loss = torch.tensor(0, device=real_patch.device)

            lpips_loss += patch_lpips_loss

    return lpips_loss / real_patches.size(2)  # Normalize by the number of patches


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        # grad_output_norm = torch.linalg.vector_norm(
        #     grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        # ).mean()
        grad_output_norm = torch.norm(grad_output).mean().item()
        # nccl over all nodes
        grad_output_norm = avg_scalar_over_nodes(
            grad_output_norm, device=grad_output.device
        )

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)


@torch.no_grad()
def avg_scalar_over_nodes(value: float, device):
    value = torch.tensor(value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


def gan_disc_loss(real_preds, fake_preds, disc_type="bce"):
    if disc_type == "bce":
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_preds, torch.ones_like(real_preds)
        )
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_preds, torch.zeros_like(fake_preds)
        )
        # eval its online performance
        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

    if disc_type == "hinge":
        real_loss = nn.functional.relu(1 - real_preds).mean()
        fake_loss = nn.functional.relu(1 + fake_preds).mean()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

    return (real_loss + fake_loss) * 0.5, avg_real_preds, avg_fake_preds, acc


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5
