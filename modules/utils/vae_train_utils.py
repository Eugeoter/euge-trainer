import torch
import torch.nn.functional as F

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
