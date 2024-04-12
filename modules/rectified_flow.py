import torch


def logit_normal(mu, sigma, shape, device):
    z = torch.randn(*shape, device=device)
    z = mu + sigma * z
    t = torch.sigmoid(z)
    return t
