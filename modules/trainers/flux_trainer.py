import torch
import re
from typing import Union, List, Any, Optional, Literal
from transformers import CLIPTokenizer, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler
from .sd15_trainer import SD15Trainer
from ..train_state.flux_train_state import FluxTrainState
from ..datasets.t2i_dataset import T2IDataset
from ..pipelines.flux_pipeline import FluxPipeline
from ..utils import flux_train_utils, flux_model_utils, sd15_model_utils
from ..models.flux.flux_models import Flux

CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class FluxTrainer(SD15Trainer):
    pipeline_class = FluxPipeline
    train_state_class = FluxTrainState
    dataset_class = T2IDataset
    nnet_class = Flux

    t5xxl_max_token_length: int = None  # None for auto
    clip_l_model_name_or_path: str = None
    t5xxl_model_name_or_path: str = None

    tokenizer1: CLIPTokenizer
    tokenizer2: T5TokenizerFast

    cpu_offload_checkpointing: bool = False

    disable_mmap_load_safetensors: bool = False
    weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"] = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29

    guidance_scale: float = 3.5
    timestep_sampler_type: Literal['uniform', 'sigma', 'sigmoid'] = 'sigma'
    sigmoid_scale: float = 1.0
    model_prediction_type: Literal['raw', 'additive', 'sigma_scaled'] = 'sigma_scaled'
    discrete_flow_shift: float = 3.0

    double_blocks_to_swap: int = None
    single_blocks_to_swap: int = None
    apply_t5_attn_mask: bool = False
    masked_loss: bool = False

    def _setup_basic(self):
        super()._setup_basic()

        self.flux_name = 'schnell' if 'schnell' in self.pretrained_model_name_or_path else "dev"
        self.logger.info(f"flux name: {self.flux_name}")
        if self.t5xxl_max_token_length is None:
            self.logger.info(f"max token length of T5XXL is set to: {self.t5xxl_max_token_length}")
            self.t5xxl_max_token_length = 256 if self.flux_name == 'schnell' else 512

        self.timestep_sampling = self.timestep_sampler_type  # adapt to kohya-ss's code

    def load_diffusion_model(self):
        clip_l = flux_model_utils.load_clip_l(self.clip_l_model_name_or_path, self.weight_dtype, "cpu")
        t5xxl = flux_model_utils.load_t5xxl(self.t5xxl_model_name_or_path, self.weight_dtype, "cpu")
        flux = flux_model_utils.load_flow_model(self.flux_name, self.pretrained_model_name_or_path, self.weight_dtype, "cpu", nnet_class=self.nnet_class)
        import json
        import os
        with open(r"flux.json", "w") as f:
            sd = {k: v.shape for k, v in flux.state_dict().items()}
            json.dump(sd, f)
            print(f"flux state dict to: {os.path.abspath(f.name)}")
        return {
            'nnet': flux,
            'text_encoder1': clip_l,
            'text_encoder2': t5xxl,
        }

    def load_vae_model(self):
        vae = flux_model_utils.load_ae(self.flux_name, self.vae_model_name_or_path, self.weight_dtype, "cpu")
        return {'vae': vae}

    def load_tokenizer_model(self):
        tokenizer1 = sd15_model_utils.load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=self.tokenizer_cache_dir)
        tokenizer2 = sd15_model_utils.load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=self.tokenizer_cache_dir)
        return {'tokenizer1': tokenizer1, 'tokenizer2': tokenizer2}

    def setup_nnet_params(self):
        self.is_swapping_blocks = self.double_blocks_to_swap is not None or self.single_blocks_to_swap is not None
        if self.is_swapping_blocks:
            # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
            # This idea is based on 2kpr's great work. Thank you!
            self.logger.info(
                f"enable block swap: double_blocks_to_swap={self.double_blocks_to_swap}, single_blocks_to_swap={self.single_blocks_to_swap}"
            )
            self.nnet.enable_block_swap(self.double_blocks_to_swap, self.single_blocks_to_swap)

        training_models = []
        params_to_optimize = []

        learning_rate_nnet = self.learning_rate_nnet or self.learning_rate
        # self.logger.debug(f"learning_rate_nnet: {learning_rate_nnet}")
        train_nnet = self.train_nnet and (learning_rate_nnet > 0 or self.block_lrs is not None)
        if train_nnet:
            if self.gradient_checkpointing:
                self.nnet.enable_gradient_checkpointing(self.cpu_offload_checkpointing)
            training_models.append(self.nnet)
            if self.nnet_trainable_params:  # only train specific parameters
                self.logger.info(f"filtering trainable parameters of nnet by patterns: {self.nnet_trainable_params}")
                if isinstance(self.nnet_trainable_params, (str, re.Pattern)):
                    self.nnet_trainable_params = [self.nnet_trainable_params]
                self.nnet_trainable_params = [re.compile(p) for p in self.nnet_trainable_params]
                nnet_params = []
                self.nnet.requires_grad_(True)
                for name, params in self.nnet.named_parameters():
                    if any(pattern.match(name) for pattern in self.nnet_trainable_params):
                        params.requires_grad = True
                        nnet_params.append(params)
                    else:
                        params.requires_grad = False
                params_to_optimize.append({"params": nnet_params, "lr": learning_rate_nnet})
            elif self.block_lrs is not None:
                self.logger.info(f"applying block learning rates to nnet: {self.block_lrs}")
                self.nnet.requires_grad_(True)
                params_to_optimize = self.setup_block_lrs()
            else:
                self.logger.info(f"training all parameters of nnet")
                self.nnet.requires_grad_(True)
                params_to_optimize.append({"params": list(self.nnet.parameters()), "lr": learning_rate_nnet})
        else:
            self.nnet.requires_grad_(False)

        if self.full_fp16 or self.full_bf16:
            self.nnet.to(self.weight_dtype)
        self.nnet = self._prepare_one_model(self.nnet, train=train_nnet, name='nnet', transform_model_if_ddp=True, device_placement=[not self.is_swapping_blocks])
        if self.is_swapping_blocks:
            self.accelerator.unwrap_model(self.nnet).move_to_device_except_swap_blocks(self.accelerator.device)  # reduce peak memory usage
            self.accelerator.unwrap_model(self.nnet).prepare_block_swap_before_forward()
        self.train_nnet = train_nnet
        self.learning_rate_nnet = learning_rate_nnet

        return training_models, params_to_optimize

    def setup_text_encoder_params(self):
        training_models = []
        params_to_optimize = []

        (
            training_models_,
            params_to_optimize_,
            self.text_encoder1,
            self.train_text_encoder1,
            self.learning_rate_te1
        ) = self._setup_one_text_encoder_params(
            self.text_encoder1,
            self.learning_rate_te1,
            name="text_encoder1"
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)

        # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
        if self.train_text_encoder1:
            self.text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
            self.text_encoder1.text_model.final_layer_norm.requires_grad_(False)

        (
            training_models_,
            params_to_optimize_,
            self.text_encoder2,
            self.train_text_encoder2,
            self.learning_rate_te2
        ) = self._setup_one_text_encoder_params(
            self.text_encoder2,
            self.learning_rate_te2,
            name="text_encoder2"
        )
        training_models.extend(training_models_)
        params_to_optimize.extend(params_to_optimize_)
        return training_models, params_to_optimize

    def _setup_diffusion(self):
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=self.discrete_flow_shift)

    # def get_train_state(self):
    #     return self.train_state_class.from_config(
    #         self.config,
    #         self,
    #         self.accelerator,
    #         train_dataset=self.train_dataset,
    #         valid_dataset=self.valid_dataset,
    #         pipeline_class=self.pipeline_class,
    #         optimizer=self.optimizer,
    #         lr_scheduler=self.lr_scheduler,
    #         train_dataloader=self.train_dataloader,
    #         valid_dataloader=self.valid_dataloader,
    #         save_dtype=self.save_dtype,
    #         nnet=self.nnet,
    #         text_encoder=[self.text_encoder1, self.text_encoder2],
    #         tokenizer=[self.tokenizer1, self.tokenizer2],
    #         noise_scheduler=self.noise_scheduler,
    #         vae=self.vae,
    #         train_nnet=self.train_nnet,
    #         train_text_encoder=[self.train_text_encoder1, self.train_text_encoder2],
    #     )

    def train_step(self, batch) -> float:
        if batch.get("latents") is not None:
            latents = batch["latents"].to(self.device, dtype=self.weight_dtype)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.device, dtype=self.vae.dtype))

        text_encoder_conds = self.encode_caption(batch['captions'])

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            self, self.noise_scheduler, latents, noise, self.accelerator.device, self.weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_model_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_model_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=self.accelerator.device)

        # get guidance
        guidance_vec = torch.full((bsz,), float(self.guidance_scale), device=self.accelerator.device)

        # call model
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not self.apply_t5_attn_mask:
            t5_attn_mask = None

        with self.accelerator.autocast():
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
            model_pred = self.nnet(
                img=packed_noisy_model_input,
                img_ids=img_ids,
                txt=t5_out,
                txt_ids=txt_ids,
                y=l_pooled,
                timesteps=timesteps / 1000,
                guidance=guidance_vec,
                txt_attention_mask=t5_attn_mask,
            )

        # unpack latents
        model_pred = flux_model_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(self, model_pred, noisy_model_input, sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        loss = self.get_loss(model_pred, target, timesteps, batch)
        if weighting is not None:
            loss = loss * weighting
        # if self.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
        #     loss = advanced_train_utils.apply_masked_loss(loss, batch)
        loss = loss.mean()
        return loss

    def encode_caption(self, captions):
        input_ids_list = [[ids[0] for ids in flux_train_utils.tokenize([self.tokenizer1, self.tokenizer2], caption, self.t5xxl_max_token_length)] for caption in captions]  # remove batch dimension
        input_ids_list = [torch.stack([input_ids[i] for input_ids in input_ids_list]).to(self.accelerator.device) for i in range(len(input_ids_list[0]))]  # stack to make a list of tensors
        with torch.no_grad():
            text_encoder_conds = flux_train_utils.encode_tokens([self.text_encoder1, self.text_encoder2], input_ids_list, apply_t5_attn_mask=self.apply_t5_attn_mask)
            if self.full_fp16:
                text_encoder_conds = [c.to(self.weight_dtype) for c in text_encoder_conds]
        return text_encoder_conds
