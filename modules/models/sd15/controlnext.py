from typing import Union

import torch
import einops
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.dim ** -0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape_as(x)
        out = self.to_out(out)

        return out


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, context):
        concat = torch.cat((x, context), dim=1)
        concat = concat.permute(0, 2, 3, 1)
        concat = self.linear(concat)
        concat = concat.permute(0, 3, 1, 2)
        return concat


class ControlNeXtModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        time_embed_dim=256,
        in_channels=[128, 128],
        out_channels=[128, 256],
        groups=[4, 8],
        controlnext_scale=1.,
        learnable_scale=False,
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )

        self.down_res = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                ),
            )
            self.down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )

        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=320,
                kernel_size=1,
                stride=1,
            ))

        self.scale = controlnext_scale if not learnable_scale else nn.Parameter(torch.tensor(controlnext_scale))

        self.cross_attn = CrossAttention(320)
        self.concat_linear = ConcatLinear(640, 320)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
    ):

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        sample = self.embedding(sample)

        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)

        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)

        return {
            'out': sample,
            'scale': self.scale,
        }
