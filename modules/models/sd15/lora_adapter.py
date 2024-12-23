import torch
import torch.nn as nn
from typing import List
from waifuset import logging


class LayerWiseMultiLoRAAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        up_weights: List[torch.Tensor],
        down_weights: List[torch.Tensor],
        alphas: List[torch.Tensor],
        lora_name: str,
        init_w0: float = None,
        init_w1: List[float] = None,
    ):
        super(LayerWiseMultiLoRAAdapter, self).__init__()
        self.module = module
        self.up_weights, self.down_weights, self.alphas = up_weights, down_weights, alphas
        self.lora_name = lora_name

        init_w0 = init_w0 if init_w0 is not None else torch.tensor(1.0)
        init_w1 = init_w1 if init_w1 is not None else [torch.tensor(1.0) for _ in range(len(up_weights))]

        self.num_loras = len(up_weights)
        assert len(down_weights) == self.num_loras and len(alphas) == self.num_loras, f"Number of UDA parameters do not match (lora_name={lora_name})"
        assert len(init_w1) == self.num_loras, f"Number of initial LoRA ratios do not match number of UDAs (lora_name={lora_name})"

        self.w0 = nn.Parameter(torch.tensor(init_w0))
        self.w1 = nn.ParameterList([nn.Parameter(torch.tensor(init_w1[i])) for i in range(self.num_loras)])

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
        self.w0.data = self.w0.data.to(*args, **kwargs)
        for param in self.w1:
            param.data = param.data.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x):
        # Forward pass through the model
        model_pred = self.w0 * self.module(x)

        # Forward pass through the LoRAs
        lora_pred = torch.zeros_like(model_pred)
        for i in range(self.num_loras):

            up_weight, down_weight, alpha = self.up_weights[i], self.down_weights[i], self.alphas[i]
            if up_weight is None or down_weight is None or alpha is None:
                continue
            scale = alpha / down_weight.size(0)

            if isinstance(self.module, nn.Linear):
                delta_theta = (up_weight @ down_weight) * scale

                # Decompose linear operation
                lora_pred += self.w1[i] * nn.functional.linear(x, delta_theta)

            elif isinstance(self.module, nn.Conv2d):
                if up_weight.dim() == 4:
                    # For Conv2d layers with 3x3 kernels
                    delta_theta = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3),
                        up_weight,
                        bias=None,
                        stride=1,
                        padding='same'
                    ).permute(1, 0, 2, 3) * scale
                else:
                    # For Conv2d layers with 1x1 kernels
                    delta_theta = (
                        (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )

                # Decompose convolution operation
                lora_pred += self.w1[i] * nn.functional.conv2d(x, delta_theta, stride=self.module.stride, padding=self.module.padding)

            else:
                pass

        output = model_pred + lora_pred
        return output

    def merged_weight(self):
        psi = self.module.weight * self.w0
        for i in range(self.num_loras):
            up_weight, down_weight, alpha = self.up_weights[i], self.down_weights[i], self.alphas[i]
            if up_weight is None or down_weight is None or alpha is None:
                continue
            scale = alpha / down_weight.size(0)

            if isinstance(self.module, nn.Linear):
                delta_theta = (up_weight @ down_weight) * scale
                psi += self.w1[i] * delta_theta
            elif isinstance(self.module, nn.Conv2d):
                if up_weight.dim() == 4:
                    # For Conv2d layers with 3x3 kernels
                    delta_theta = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3),
                        up_weight,
                        bias=None,
                        stride=1,
                        padding='same'
                    ).permute(1, 0, 2, 3) * scale
                else:
                    # For Conv2d layers with 1x1 kernels
                    delta_theta = (
                        (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                psi += self.w1[i] * delta_theta
            else:
                raise ValueError(f"LoRA module must be either nn.Linear or nn.Conv2d, got {type(self.module)}")
        return psi

    def merged_module(self):
        # merge loras into the module with ratios
        if isinstance(self.module, nn.Linear):
            module = nn.Linear(self.module.in_features, self.module.out_features, bias=self.module.bias is not None)
            module.weight.data = self.merged_weight()
            if self.module.bias is not None:
                module.bias.data = self.module.bias
        elif isinstance(self.module, nn.Conv2d):
            module = nn.Conv2d(
                self.module.in_channels,
                self.module.out_channels,
                self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
                bias=self.module.bias is not None,
            )
            module.weight.data = self.merged_weight()
            if self.module.bias is not None:
                module.bias.data = self.module.bias
        else:
            raise ValueError(f"LoRA module must be either nn.Linear or nn.Conv2d, got {type(self.module)}")
        return module


class ElementWiseMultiLoRAAdapter(LayerWiseMultiLoRAAdapter):
    def __init__(
        self,
        module: nn.Module,
        up_weights: List[torch.Tensor],
        down_weights: List[torch.Tensor],
        alphas: List[torch.Tensor],
        lora_name: str,
        init_w0: float = None,
        init_w1: List[float] = None,
    ):
        super(LayerWiseMultiLoRAAdapter, self).__init__()
        self.module = module
        self.up_weights, self.down_weights, self.alphas = up_weights, down_weights, alphas
        self.lora_name = lora_name

        init_w0 = init_w0 if init_w0 is not None else torch.tensor(1.0)
        init_w1 = init_w1 if init_w1 is not None else [torch.tensor(1.0) for _ in range(len(up_weights))]

        self.num_loras = len(up_weights)
        assert len(down_weights) == self.num_loras and len(alphas) == self.num_loras, f"Number of UDA parameters do not match (lora_name={lora_name})"
        assert len(init_w1) == self.num_loras, f"Number of initial LoRA ratios do not match number of UDAs (lora_name={lora_name})"

        self.w0 = nn.Parameter(torch.ones_like(module.weight) * init_w0)
        self.w1 = nn.ParameterList([nn.Parameter(torch.ones((up_weights[i].size(0), down_weights[i].size(1))) * init_w1[i]) for i in range(self.num_loras)])

    def forward(self, x):
        # Forward pass through the model

        if isinstance(self.module, nn.Linear):
            model_pred = nn.functional.linear(x, self.module.weight * self.w0, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            model_pred = nn.functional.conv2d(x, self.module.weight * self.w0, self.module.bias, stride=self.module.stride, padding=self.module.padding)

        # Forward pass through the LoRAs
        lora_pred = torch.zeros_like(model_pred)
        for i in range(self.num_loras):

            up_weight, down_weight, alpha = self.up_weights[i], self.down_weights[i], self.alphas[i]
            if up_weight is None or down_weight is None or alpha is None:
                continue
            scale = alpha / down_weight.size(0)

            if isinstance(self.module, nn.Linear):
                delta_theta = (up_weight @ down_weight) * scale

                delta_psi = self.w1[i] * delta_theta

                # Decompose linear operation
                lora_pred += nn.functional.linear(x, delta_psi)

            elif isinstance(self.module, nn.Conv2d):
                if up_weight.dim() == 4:
                    # For Conv2d layers with 3x3 kernels
                    delta_theta = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3),
                        up_weight,
                        bias=None,
                        stride=1,
                        padding='same'
                    ).permute(1, 0, 2, 3) * scale
                else:
                    # For Conv2d layers with 1x1 kernels
                    delta_theta = (
                        (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                delta_psi = (self.w1[i] * delta_theta.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)

                # Decompose convolution operation
                lora_pred += nn.functional.conv2d(x, delta_psi, stride=self.module.stride, padding=self.module.padding)

            else:
                pass

        output = model_pred + lora_pred
        return output
