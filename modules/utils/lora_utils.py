import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import copy
from typing import Literal, Dict, Any, List, Tuple
from waifuset import logging
from ..models.sd15.lora_adapter import ElementWiseMultiLoRAAdapter

logger = logging.get_logger("merge lora")


UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"] + [ElementWiseMultiLoRAAdapter]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"] + [ElementWiseMultiLoRAAdapter]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"] + ["CLIPSdpaAttention"] + [ElementWiseMultiLoRAAdapter]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

# SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

LORA_UNET_IN = r"lora_unet_down_blocks_"
LORA_UNET_IN_0 = r"lora_unet_down_blocks_0_"
LORA_UNET_IN_1 = r"lora_unet_down_blocks_1_"
LORA_UNET_IN_2 = r"lora_unet_down_blocks_2_"
# LORA_UNET_IN_3 = r"lora_unet_down_blocks_3_"
LORA_UNET_MID = r"lora_unet_mid_block_(?:attentions|resnets)_"
LORA_UNET_OUT = r"lora_unet_up_blocks_"
# LORA_UNET_OUT_0 = r"lora_unet_up_blocks_0_"
LORA_UNET_OUT_1 = r"lora_unet_up_blocks_1_"
LORA_UNET_OUT_2 = r"lora_unet_up_blocks_2_"
LORA_UNET_OUT_3 = r"lora_unet_up_blocks_3_"
LORA_TEXT_ENCODER = r"lora_te_"

UNET_IN = r"down_blocks\."
UNET_IN_0 = r"down_blocks\.0\."
UNET_IN_1 = r"down_blocks\.1\."
UNET_IN_2 = r"down_blocks\.2\."
# UNET_IN_3 = r"down_blocks\.3\."
UNET_MID = r"mid_block\.(?:attentions|resnets)\."
UNET_OUT = r"up_blocks\."
# UNET_OUT_0 = r"up_blocks\.0\."
UNET_OUT_1 = r"up_blocks\.1\."
UNET_OUT_2 = r"up_blocks\.2\."
UNET_OUT_3 = r"up_blocks\.3\."
TEXT_ENCODER = r"text_model\."

# [te, in0, in1, in2, mid, out1, out2, out3]


# def set_ratios_to_warped_model(modules, model_type: Literal['sdxl', 'sd15'] = 'sd15', ratios: torch.Tensor = None):
#     if model_type == 'sdxl':
#         modules = (
#             nnet,
#             text_encoder,
#             text_encoder_2,
#         ) = modules['nnet'], modules['text_encoder1'], modules['text_encoder2']
#     elif model_type == 'sd15':
#         modules = (
#             nnet,
#             text_encoder,
#         ) = modules['nnet'], modules['text_encoder']
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

#     for i, root_module in enumerate(modules):
#         if i >= 1:
#             if model_type == 'sdxl':
#                 if i == 1:
#                     prefix = LORA_PREFIX_TEXT_ENCODER1
#                 else:
#                     prefix = LORA_PREFIX_TEXT_ENCODER2
#             else:
#                 prefix = LORA_PREFIX_TEXT_ENCODER
#             target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE
#         else:
#             prefix = LORA_PREFIX_UNET
#             target_replace_modules = (
#                 UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
#             )

#         for name, module in root_module.named_modules():
#             if module.__class__.__name__ in target_replace_modules:
#                 for child_name, child_module in module.named_modules():
#                     if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
#                         lora_name = prefix + "." + name + "." + child_name
#                         lora_name = lora_name.replace(".", "_")
#                         ratio_index = get_ratio_index_from_lora_key(lora_name)
#                         child_module.set_ratio(ratios[ratio_index])


def make_lora_name_to_module_map(modules, model_type: Literal['sdxl', 'sd15'] = 'sd15', debug_te=False):
    name_to_module = {}
    for i, root_module in enumerate(modules):
        if i >= 1:  # te
            if model_type == 'sdxl':  # sdxl
                if i == 1:
                    prefix = LORA_PREFIX_TEXT_ENCODER1
                else:
                    prefix = LORA_PREFIX_TEXT_ENCODER2
            else:  # sd15
                prefix = LORA_PREFIX_TEXT_ENCODER
            target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE
        else:  # unet
            prefix = LORA_PREFIX_UNET
            target_replace_modules = (
                UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"enter {logging.green(module.__class__.__name__)}")
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module
                        if debug_te and "lora_te" in prefix:
                            logger.debug(f"    found {logging.green(child_module.__class__.__name__)} | {name + '.' + child_name}")
            else:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"skip {logging.red(module.__class__.__name__)}")
    return name_to_module


def make_lora_name_to_lora_wrapper_map(modules, model_type: Literal['sdxl', 'sd15'] = 'sd15', debug_te=False):
    name_to_module = {}
    for i, root_module in enumerate(modules):
        if i >= 1:  # te
            if model_type == 'sdxl':  # sdxl
                if i == 1:
                    prefix = LORA_PREFIX_TEXT_ENCODER1
                else:
                    prefix = LORA_PREFIX_TEXT_ENCODER2
            else:  # sd15
                prefix = LORA_PREFIX_TEXT_ENCODER
            target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE
        else:  # unet
            prefix = LORA_PREFIX_UNET
            target_replace_modules = (
                UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"enter {logging.green(module.__class__.__name__)}")
                for child_name, child_module in module.named_modules():
                    if "LoRAAdapter" in child_module.__class__.__name__:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module
                        if debug_te and "lora_te" in prefix:
                            logger.debug(f"    found {logging.green(child_module.__class__.__name__)} | {name + '.' + child_name}")
            else:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"skip {logging.red(module.__class__.__name__)}")
    return name_to_module


def make_lora_name_to_module_name_map(modules, model_type: Literal['sdxl', 'sd15'] = 'sd15', debug_te=False):
    lora_name_to_module_name = {}
    for i, root_module in enumerate(modules):
        if i >= 1:  # te
            if model_type == 'sdxl':  # sdxl
                if i == 1:
                    prefix = LORA_PREFIX_TEXT_ENCODER1
                else:
                    prefix = LORA_PREFIX_TEXT_ENCODER2
            else:  # sd15
                prefix = LORA_PREFIX_TEXT_ENCODER
            target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE
        else:  # unet
            prefix = LORA_PREFIX_UNET
            target_replace_modules = (
                UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"enter {logging.green(module.__class__.__name)}")
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d" or "loradapter" in child_module.__class__.__name__.lower():
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora_name_to_module_name[lora_name] = name + "." + child_name
                        if debug_te and "lora_te" in prefix:
                            logger.debug(f"    found {logging.green(child_module.__class__.__name)} | {name + '.' + child_name}")
            else:
                if debug_te and "lora_te" in prefix:
                    logger.debug(f"skip {logging.red(module.__class__.__name)}")
    return lora_name_to_module_name


def make_lora_name_to_weight_shape_map(
    lora_state_dicts: List[Dict[str, torch.Tensor]],
    model_type: Literal['sdxl', 'sd15'] = 'sd15',
) -> Dict[str, Tuple[int, int]]:
    lora_name_to_weight_shape = {}
    for lora_state_dict in lora_state_dicts:
        for key in lora_state_dict.keys():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                # logging.debug(f"lora_name: {logging.yellow(lora_name)}")
                if lora_name in lora_name_to_weight_shape:  # already processed
                    # logging.debug(f"  skip {logging.red(lora_name)}")
                    continue
                down_key = key
                up_key = key.replace("lora_down", "lora_up")

                # weight_adjustment = up_weight @ down_weight, where up_weight is (up_dim, rank) and down_weight is (rank, down_dim)
                # weight_shape = (up_dim, down_dim)
                weight_shape = lora_state_dict[up_key].size(0), lora_state_dict[down_key].size(1)
                lora_name_to_weight_shape[lora_name] = weight_shape
                # logging.debug(f"  process {logging.green(lora_name)}: {weight_shape}")
    return lora_name_to_weight_shape


def merge_loras_to_model(
    models: Dict[str, Any],
    lora_state_dicts,
    lora_strength,
    model_type: Literal['sdxl', 'sd15'] = 'sd15',
    merge_device='cpu',
    merge_dtype=torch.float32,
    name_to_module=None,
    inplace=False,
    verbose=False,
) -> Dict[str, Any]:
    r"""
    Merge LoRA weights to the model.

    @param models: A dictionary that stores the model components, e.g., nnet, text encoders, etc.
    @param lora_state_dicts: LoRA weights stored in a list of state dicts.
    @param lora_ratios: Ratios for LoRA weights.
    @param model_type: The model type, either 'sdxl' or 'sd15'.
    @param merge_device: The device to merge the LoRA weights.
    @param merge_dtype: The data type to merge the LoRA weights.

    @return: The updated models.
    """
    if not inplace:
        models = copy.deepcopy(models)

    # Determine model type from sd1.5 and sdxl
    if model_type == 'sdxl':
        modules = (
            nnet,
            text_encoder,
            text_encoder_2,
        ) = models['nnet'], models['text_encoder1'], models['text_encoder2']
    elif model_type == 'sd15':
        modules = (
            nnet,
            text_encoder,
        ) = models['nnet'], models['text_encoder']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create module map
    name_to_module = name_to_module or make_lora_name_to_module_map(modules, model_type)

    for lora_state_dict, ratios in logger.tqdm(zip(lora_state_dicts, lora_strength), desc="lora", position=0, total=len(lora_state_dicts), leave=True, disable=not verbose):
        for key in logger.tqdm(lora_state_dict.keys(), desc="merge", position=1, leave=True, disable=not verbose):
            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                # match ratio
                if isinstance(ratios, (int, float)):
                    ratio = ratios
                elif isinstance(ratios, dict):
                    for k, v in ratios.items():
                        if key.startswith(k):
                            ratio = v
                            break
                    else:
                        continue
                elif isinstance(ratios, list):
                    if not len(ratios) == 8:
                        raise ValueError(f"ratio length must be 8, got {len(ratios)}")

                    ratio_index = get_ratio_index_from_lora_key(key)
                    ratio = ratios[ratio_index]

                if ratio == 0:
                    continue

                # find original module for this lora
                module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                if module_name not in name_to_module:
                    raise ValueError(f"LoRA Module name {module_name} not found in module map")
                module = name_to_module[module_name]
                # print(f"apply {key} to {module}")

                down_weight = lora_state_dict[key].to(merge_device, merge_dtype)
                up_weight = lora_state_dict[up_key].to(merge_device, merge_dtype)

                dim = down_weight.size()[0]
                alpha = lora_state_dict.get(alpha_key, dim).to(merge_device, merge_dtype)
                scale = alpha / dim

                # W <- W + U * D
                weight = module.weight
                weight = weight.to(merge_device, merge_dtype)
                # print(module_name, down_weight.size(), up_weight.size())
                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                module.weight = torch.nn.Parameter(weight.to(module.weight.device, dtype=module.weight.dtype))

    return models


def get_module_by_name(models, module_name):
    for model in models.values():
        modules = dict(model.named_modules())
        if module_name in modules:
            return modules[module_name]
    else:
        raise ValueError(f"Module {module_name} not found in {modules.keys()}")
    return None


def set_module_by_name(models, module_name, new_module):
    for model in models.values():
        modules = dict(model.named_modules())
        if module_name in modules:
            parent_module = get_parent_module(model, module_name)
            child_name = module_name.split('.')[-1]
            setattr(parent_module, child_name, new_module)
            break
    else:
        raise ValueError(f"Module {module_name} not found in {modules.keys()}")


def get_parent_module(root_module, module_name):
    names = module_name.split('.')
    parent = root_module
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent


def get_ratio_index_from_lora_key(key):
    if re.search(LORA_TEXT_ENCODER, key):
        return 0
    elif re.search(LORA_UNET_IN_0, key):
        return 1
    elif re.search(LORA_UNET_IN_1, key):
        return 2
    elif re.search(LORA_UNET_IN_2, key):
        return 3
    elif re.search(LORA_UNET_MID, key):
        return 4
    elif re.search(LORA_UNET_OUT_1, key):
        return 5
    elif re.search(LORA_UNET_OUT_2, key):
        return 6
    elif re.search(LORA_UNET_OUT_3, key):
        return 7
    else:
        raise ValueError(f"Unknown key: {key}")


def get_ratio_index_from_module_key(key):
    if re.search(TEXT_ENCODER, key):
        return 0
    elif re.search(UNET_IN_0, key):
        return 1
    elif re.search(UNET_IN_1, key):
        return 2
    elif re.search(UNET_IN_2, key):
        return 3
    elif re.search(UNET_MID, key):
        return 4
    elif re.search(UNET_OUT_1, key):
        return 5
    elif re.search(UNET_OUT_2, key):
        return 6
    elif re.search(UNET_OUT_3, key):
        return 7
    else:
        raise ValueError(f"Unknown key: {key}")


def wrap_lora_to_model(
    models: Dict[str, nn.Module],
    lora_state_dict: Dict[str, torch.Tensor],
    lora_ratios: Dict[str, torch.Tensor],  # {lora_name: torch.Tensor}
    model_type: Literal['sdxl', 'sd15'] = 'sd15',
    merge_device='cpu',
    merge_dtype=torch.float32,
    name_to_module=None,
    name_to_module_name=None,
    inplace=False,
    verbose=False,
) -> Dict[str, Any]:
    r"""
    Merge LoRA weights to the model.

    @param models: A dictionary that stores the model components, e.g., nnet, text encoders, etc.
    @param lora_state_dicts: LoRA weights stored in a list of state dicts.
    @param lora_ratios: Ratios for LoRA weights.
    @param model_type: The model type, either 'sdxl' or 'sd15'.
    @param merge_device: The device to merge the LoRA weights.
    @param merge_dtype: The data type to merge the LoRA weights.

    @return: The updated models.
    """
    if not inplace:
        models = copy.deepcopy(models)

    # Determine model type from sd1.5 and sdxl
    if model_type == 'sdxl':
        modules = (
            nnet,
            text_encoder,
            text_encoder_2,
        ) = models['nnet'], models['text_encoder1'], models['text_encoder2']
    elif model_type == 'sd15':
        modules = (
            nnet,
            text_encoder,
        ) = models['nnet'], models['text_encoder']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    name_to_module = name_to_module or make_lora_name_to_module_map(modules, model_type)
    name_to_module_name = name_to_module_name or make_lora_name_to_module_name_map(modules, model_type)

    for key in logger.tqdm(lora_state_dict.keys(), desc="merge lora", position=1, leave=True, disable=not verbose):
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            # Find the original module
            lora_name = ".".join(key.split(".")[:-2])

            if lora_name not in name_to_module:
                raise ValueError(f"LoRA module name {lora_name} not found in module map")
            if lora_name not in name_to_module_name:
                raise ValueError(f"LoRA module name {lora_name} not found in module name map")
            if lora_name not in lora_ratios:
                raise ValueError(f"Lora module name {lora_name} not found in lora ratios. Lora ratios: {lora_ratios}")

            module = name_to_module[lora_name]
            module_name = name_to_module_name[lora_name]
            ratio = lora_ratios[lora_name]

            # Load LoRA weights
            down_weight = lora_state_dict[key]
            up_weight = lora_state_dict[up_key]
            alpha = lora_state_dict.get(alpha_key, down_weight.size(0))

            # Wrap the module with LoRAAdapter
            wrapped_module = ElementWiseMultiLoRAAdapter(module, up_weight, down_weight, alpha, ratio)

            # Set the wrapped module to the model
            set_module_by_name(models, module_name, wrapped_module)

    return models


def wrap_loras_to_model(
    models: Dict[str, nn.Module],
    lora_state_dicts: List[Dict[str, torch.Tensor]],
    init_w0: float = None,
    init_w1: List[float] = None,
    model_type: Literal['sdxl', 'sd15'] = 'sd15',
    lora_name_to_module=None,
    lora_name_to_module_name=None,
    lora_wrapper_class=ElementWiseMultiLoRAAdapter,
    inplace=False,
    verbose=False,
) -> Dict[str, nn.Module]:
    if not inplace:
        models = copy.deepcopy(models)

    # Determine model type from sd1.5 and sdxl
    if model_type == 'sdxl':
        modules = (
            nnet,
            text_encoder,
            text_encoder_2,
        ) = models['nnet'], models['text_encoder1'], models['text_encoder2']
    elif model_type == 'sd15':
        modules = (
            nnet,
            text_encoder,
        ) = models['nnet'], models['text_encoder']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    lora_name_to_module = lora_name_to_module or make_lora_name_to_module_map(modules, model_type)
    lora_name_to_module_name = lora_name_to_module_name or make_lora_name_to_module_name_map(modules, model_type)

    for lora_name, module in logger.tqdm(lora_name_to_module.items(), desc="merge lora", position=1, leave=True, disable=not verbose):
        udas = []
        for state_dict in lora_state_dicts:
            alpha_key = lora_name + '.alpha'
            up_key = lora_name + '.lora_up.weight'
            down_key = lora_name + '.lora_down.weight'

            if down_key not in state_dict:
                up_weight, down_weight, alpha = None, None, None
            else:
                # Load LoRA weights
                down_weight = state_dict[down_key]
                up_weight = state_dict[up_key]
                alpha = state_dict.get(alpha_key, down_weight.size(0))
            udas.append((up_weight, down_weight, alpha))

        if not isinstance(module, lora_wrapper_class) and any(lora_name + '.lora_down.weight' in state_dict for state_dict in lora_state_dicts):
            wrapped_module = lora_wrapper_class(
                module=module,
                up_weights=[uda[0] for uda in udas],
                down_weights=[uda[1] for uda in udas],
                alphas=[uda[2] for uda in udas],
                lora_name=lora_name,
                init_w0=init_w0,
                init_w1=init_w1,
            )
            if lora_name not in lora_name_to_module_name:
                raise ValueError(f"LoRA module name {lora_name} not found in module name map")
            module_name = lora_name_to_module_name[lora_name]
            set_module_by_name(models, module_name, wrapped_module)
        else:
            continue

    return models


def set_ratios_to_warped_model(
    models,
    lora_name_to_ratios: Dict[str, torch.Tensor],
    lora_name_to_module=None,
    model_type: Literal['sdxl', 'sd15'] = 'sd15',
    inplace=False,
    verbose=False,
):
    if not inplace:
        models = copy.deepcopy(models)

    # Determine model type from sd1.5 and sdxl
    if model_type == 'sdxl':
        modules = (
            nnet,
            text_encoder,
            text_encoder_2,
        ) = models['nnet'], models['text_encoder1'], models['text_encoder2']
    elif model_type == 'sd15':
        modules = (
            nnet,
            text_encoder,
        ) = models['nnet'], models['text_encoder']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    lora_name_to_module = lora_name_to_module or make_lora_name_to_lora_wrapper_map(modules, model_type)

    for lora_name, module in logger.tqdm(lora_name_to_module.items(), desc="merge lora", position=1, leave=True, disable=not verbose):
        # if not isinstance(module, MultiLoRAAdapter):
        #     raise ValueError(f"Module {module.__class__.__name__} (lora name is {lora_name}) is not a {MultiLoRAAdapter.__name__}")

        if lora_name not in lora_name_to_ratios:
            # logging.debug(f"skip {logging.red(lora_name)}")
            continue
        # logging.debug(f"set ratios to {logging.green(lora_name)}")
        ratios: nn.ParameterList = lora_name_to_ratios[lora_name]
        try:
            module.set_ratios(ratios)
        except AttributeError:
            raise AttributeError(f"Error in setting ratios to {module.__class__.__name__}) (lora name is {lora_name})")

    return models
