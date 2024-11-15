from typing import Optional, Dict
from .flux_models import *


class FluxWithControlNeXt(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.double_blocks_to_swap = None
        self.single_blocks_to_swap = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        self.time_in.enable_gradient_checkpointing()
        self.vector_in.enable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.enable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        print(f"FLUX: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.time_in.disable_gradient_checkpointing()
        self.vector_in.disable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.disable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing disabled.")

    def enable_block_swap(self, double_blocks: Optional[int], single_blocks: Optional[int]):
        self.double_blocks_to_swap = double_blocks
        self.single_blocks_to_swap = single_blocks

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu
        if self.double_blocks_to_swap:
            save_double_blocks = self.double_blocks
            self.double_blocks = None
        if self.single_blocks_to_swap:
            save_single_blocks = self.single_blocks
            self.single_blocks = None

        self.to(device)

        if self.double_blocks_to_swap:
            self.double_blocks = save_double_blocks
        if self.single_blocks_to_swap:
            self.single_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        # move last n blocks to cpu: they are on cuda
        if self.double_blocks_to_swap:
            for i in range(len(self.double_blocks) - self.double_blocks_to_swap):
                self.double_blocks[i].to(self.device)
            for i in range(len(self.double_blocks) - self.double_blocks_to_swap, len(self.double_blocks)):
                self.double_blocks[i].to("cpu")  # , non_blocking=True)
        if self.single_blocks_to_swap:
            for i in range(len(self.single_blocks) - self.single_blocks_to_swap):
                self.single_blocks[i].to(self.device)
            for i in range(len(self.single_blocks) - self.single_blocks_to_swap, len(self.single_blocks)):
                self.single_blocks[i].to("cpu")  # , non_blocking=True)
        clean_memory_on_device(self.device)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        txt_attention_mask: Tensor | None = None,
        controls: Dict[str, torch.Tensor] | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        # print(f"img.shape before img_in: {img.shape}")
        img = self.img_in(img)

        if controls is not None:
            scale = controls['scale']
            signal = controls['out'].to(img)
            mean_latents, std_latents = torch.mean(img, dim=(1, 2), keepdim=True), torch.std(img, dim=(1, 2), keepdim=True)
            mean_control, std_control = torch.mean(signal, dim=(1, 2), keepdim=True), torch.std(signal, dim=(1, 2), keepdim=True)
            signal = (signal - mean_control) * (std_latents / (std_control + 1e-12)) + mean_latents
            img = img + signal * scale

        # print(f"img.shape after img_in: {img.shape}")
        vec = self.time_in(timestep_embedding(timesteps, 256))
        # print(f"controls.shape: {controls['out'].shape}")
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if not self.double_blocks_to_swap:
            for i, block in enumerate(self.double_blocks):
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
                # print(f"img.shape after double block {i}: {img.shape}")
        else:
            # make sure first n blocks are on cuda, and last n blocks are on cpu at beginning
            for block_idx in range(self.double_blocks_to_swap):
                block = self.double_blocks[len(self.double_blocks) - self.double_blocks_to_swap + block_idx]
                if block.parameters().__next__().device.type != "cpu":
                    block.to("cpu")  # , non_blocking=True)
                    # print(f"Moved double block {len(self.double_blocks) - self.double_blocks_to_swap + block_idx} to cpu.")

                block = self.double_blocks[block_idx]
                if block.parameters().__next__().device.type == "cpu":
                    block.to(self.device)
                    # print(f"Moved double block {block_idx} to cuda.")

            to_cpu_block_index = 0
            for block_idx, block in enumerate(self.double_blocks):
                # move last n blocks to cuda: they are on cpu, and move first n blocks to cpu: they are on cuda
                moving = block_idx >= len(self.double_blocks) - self.double_blocks_to_swap
                if moving:
                    block.to(self.device)  # move to cuda
                    # print(f"Moved double block {block_idx} to cuda.")

                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)

                if moving:
                    self.double_blocks[to_cpu_block_index].to("cpu")  # , non_blocking=True)
                    # print(f"Moved double block {to_cpu_block_index} to cpu.")
                    to_cpu_block_index += 1

        img = torch.cat((txt, img), 1)

        if not self.single_blocks_to_swap:
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
                # print(f"img.shape after single block: {img.shape}")
        else:
            # make sure first n blocks are on cuda, and last n blocks are on cpu at beginning
            for block_idx in range(self.single_blocks_to_swap):
                block = self.single_blocks[len(self.single_blocks) - self.single_blocks_to_swap + block_idx]
                if block.parameters().__next__().device.type != "cpu":
                    block.to("cpu")  # , non_blocking=True)
                    # print(f"Moved single block {len(self.single_blocks) - self.single_blocks_to_swap + block_idx} to cpu.")

                block = self.single_blocks[block_idx]
                if block.parameters().__next__().device.type == "cpu":
                    block.to(self.device)
                    # print(f"Moved single block {block_idx} to cuda.")

            to_cpu_block_index = 0
            for block_idx, block in enumerate(self.single_blocks):
                # move last n blocks to cuda: they are on cpu, and move first n blocks to cpu: they are on cuda
                moving = block_idx >= len(self.single_blocks) - self.single_blocks_to_swap
                if moving:
                    block.to(self.device)  # move to cuda
                    # print(f"Moved single block {block_idx} to cuda.")

                img = block(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)

                if moving:
                    self.single_blocks[to_cpu_block_index].to("cpu")  # , non_blocking=True)
                    # print(f"Moved single block {to_cpu_block_index} to cpu.")
                    to_cpu_block_index += 1

        img = img[:, txt.shape[1]:, ...]

        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class FluxUpper(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        self.time_in.enable_gradient_checkpointing()
        self.vector_in.enable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.enable_gradient_checkpointing()

        for block in self.double_blocks:
            block.enable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        self.time_in.disable_gradient_checkpointing()
        self.vector_in.disable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.disable_gradient_checkpointing()

        for block in self.double_blocks:
            block.disable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing disabled.")

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        txt_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)

        return img, txt, vec, pe


class FluxLower(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.out_channels = params.in_channels

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.single_blocks:
            block.enable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.single_blocks:
            block.disable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing disabled.")

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor | None = None,
        pe: Tensor | None = None,
        txt_attention_mask: Tensor | None = None,
    ) -> Tensor:
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
