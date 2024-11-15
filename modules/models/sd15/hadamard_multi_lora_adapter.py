import torch
import torch.nn as nn
from typing import List
from waifuset import logging


class HadamardMultiLoRAWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        up_weights: List[torch.Tensor],
        down_weights: List[torch.Tensor],
        alphas: List[torch.Tensor],
        lora_name: str,
        init_module_ratio: float = None,
        init_lora_ratios: List[float] = None,
    ):
        super(HadamardMultiLoRAWrapper, self).__init__()
        self.module = module
        self.up_weights, self.down_weights, self.alphas = up_weights, down_weights, alphas
        self.lora_name = lora_name

        init_module_ratio = init_module_ratio if init_module_ratio is not None else torch.tensor(1.0)
        init_lora_ratios = init_lora_ratios if init_lora_ratios is not None else [torch.tensor(1.0) for _ in range(len(up_weights))]

        self.num_loras = len(up_weights)
        assert len(down_weights) == self.num_loras and len(alphas) == self.num_loras, f"Number of UDA parameters do not match (lora_name={lora_name})"
        assert len(init_lora_ratios) == self.num_loras, f"Number of initial LoRA ratios do not match number of UDAs (lora_name={lora_name})"

        self.module_ratio = nn.Parameter(torch.ones_like(module.weight) * init_module_ratio)
        self.lora_ratios = nn.ParameterList([nn.Parameter(torch.ones((up_weights[i].size(0), down_weights[i].size(1))) * init_lora_ratios[i]) for i in range(self.num_loras)])

    @ property
    def device(self):
        return next(self.module.parameters()).device

    @ property
    def dtype(self):
        return next(self.module.parameters()).dtype

    def to(self, *args, **kwargs):
        self.up_weights = [up_weight.to(*args, **kwargs) for up_weight in self.up_weights]
        self.down_weights = [down_weight.to(*args, **kwargs) for down_weight in self.down_weights]
        self.alphas = [alpha.to(*args, **kwargs) for alpha in self.alphas]
        return super().to(*args, **kwargs)

    def forward(self, x):
        # Forward pass through the model

        if isinstance(self.module, nn.Linear):
            model_pred = nn.functional.linear(x, self.module.weight * self.module_ratio, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            model_pred = nn.functional.conv2d(x, self.module.weight * self.module_ratio, self.module.bias, stride=self.module.stride, padding=self.module.padding)

        # Forward pass through the LoRAs
        lora_pred = torch.zeros_like(model_pred)
        for i in range(self.num_loras):

            up_weight, down_weight, alpha = self.up_weights[i], self.down_weights[i], self.alphas[i]
            if up_weight is None or down_weight is None or alpha is None:
                continue
            scale = alpha / down_weight.size(0)

            if isinstance(self.module, nn.Linear):
                lora_adjustment = (up_weight @ down_weight) * scale

                delta_weight = self.lora_ratios[i] * lora_adjustment

                # Decompose linear operation
                lora_pred += nn.functional.linear(x, delta_weight)

            elif isinstance(self.module, nn.Conv2d):
                if up_weight.dim() == 4:
                    # For Conv2d layers with 3x3 kernels
                    lora_adjustment = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3),
                        up_weight,
                        bias=None,
                        stride=1,
                        padding='same'
                    ).permute(1, 0, 2, 3) * scale
                else:
                    # For Conv2d layers with 1x1 kernels
                    lora_adjustment = (
                        (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                delta_weight = (self.lora_ratios[i] * lora_adjustment.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)

                # Decompose convolution operation
                lora_pred += nn.functional.conv2d(x, delta_weight, stride=self.module.stride, padding=self.module.padding)

            else:
                pass

        output = model_pred + lora_pred

        return output

    def merged_weight(self):
        return self.module.weight * self.module_ratio + sum([self.lora_ratios[i] * (self.alphas[i] / self.down_weights[i].size(0)) * (self.up_weights[i] @ self.down_weights[i]) for i in range(self.num_loras)])

    def merged_module(self):
        if isinstance(self.module, nn.Linear):
            return nn.Linear(self.module.in_features, self.module.out_features, bias=self.module.bias is not None)
        elif isinstance(self.module, nn.Conv2d):
            return nn.Conv2d(
                self.module.in_channels,
                self.module.out_channels,
                self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
                bias=self.module.bias is not None,
            )
        else:
            raise NotImplementedError(f"Only Linear and Conv2d modules are supported, but got {type(self.module)}")
