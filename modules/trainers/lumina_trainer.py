import torch
import os
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
from .sd15_trainer import SD15Trainer
from ..datasets.lumina_dataset import LuminaDataset
from ..train_state.lumina_train_state import LuminaTrainState
from ..utils import lumina_train_utils
from ..models import lumina as lumina_models


class LuminaTrainer(SD15Trainer):
    dataset_class = LuminaDataset
    nnet_class = None  # TODO
    pipeline_class = None  # TODO
    train_state_class = LuminaTrainState

    model_type: str = "NextDiT_2B_GQA_patch2_Adaln_Refiner"
    qk_norm: bool = True

    def load_tokenizer_model(self):
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        tokenizer.padding_side = "right"
        return {'tokenizer': tokenizer}

    def load_diffusion_model(self):
        text_encoder = AutoModel.from_pretrained(
            "google/gemma-2-2b",
            torch_dtype=torch.bfloat16,
        ).cuda()
        # text_encoder = setup_lm_fsdp_sync(text_encoder)
        cap_feat_dim = text_encoder.config.hidden_size

        # Create model:
        model = lumina_models.__dict__[self.model_type](
            in_channels=16,
            qk_norm=self.qk_norm,
            cap_feat_dim=cap_feat_dim,
        )
        self.logger.info(f"DiT Parameters: {model.parameter_count():,}")
        model_patch_size = model.patch_size

        self.logger.info(f"Initializing model weights from: {self.pretrained_model_name_or_path}")
        state_dict = torch.load(
            os.path.join(
                self.pretrained_model_name_or_path,
                f"consolidated.{0:02d}-of-{1:02d}.pth",
            ),
            map_location="cpu",
        )

        size_mismatch_keys = []
        model_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape != v.shape:
                size_mismatch_keys.append(k)
        for k in size_mismatch_keys:
            del state_dict[k]
        del model_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
        del state_dict
        # assert set(missing_keys) == set(missing_keys_ema)
        # assert set(unexpected_keys) == set(unexpected_keys_ema)
        self.logger.info("Model initialization result:")
        self.logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
        self.logger.info(f"  Missing keys: {missing_keys}")
        self.logger.info(f"  Unexpected keys: {unexpected_keys}")

        return {
            'nnet': model,
            'text_encoder': text_encoder,
        }

    def load_vae_model(self):
        vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(
            self.device
        )

    def setup_text_encoder_params(self):
