import logging
import os
from collections import OrderedDict

import torch

_logger = logging.getLogger(__name__)


def _resize_pos_embed_if_needed(pos_embed, model, resize_pos_embed):
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)

    num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
    base_len = model.patch_embed.num_patches + num_prefix_tokens
    full_len = model.pos_embed.shape[1]
    has_prompt_pos = full_len > base_len

    target_pos = model.pos_embed
    if has_prompt_pos:
        target_pos = model.pos_embed[:, :base_len].detach().clone()

    if pos_embed.shape != target_pos.shape:
        pos_embed = resize_pos_embed(
            pos_embed,
            target_pos,
            num_prefix_tokens,
            model.patch_embed.grid_size,
        )

    if has_prompt_pos:
        full_pos = model.pos_embed.detach().clone()
        prompt_len = full_len - base_len
        if num_prefix_tokens:
            full_pos[:, :num_prefix_tokens] = pos_embed[:, :num_prefix_tokens]
            full_pos[:, num_prefix_tokens + prompt_len:] = pos_embed[:, num_prefix_tokens:]
        else:
            full_pos[:, prompt_len:] = pos_embed
        pos_embed = full_pos
    return pos_embed


def _filter_to_model(state_dict, model):
    model_state = model.state_dict()
    filtered = OrderedDict()
    skipped = []
    mismatched = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value.float() if torch.is_floating_point(value) else value
    return filtered, skipped, mismatched


def _load_filtered(model, state_dict, source_name):
    filtered, skipped, mismatched = _filter_to_model(state_dict, model)
    msg = model.load_state_dict(filtered, strict=False)
    _logger.info(
        "Loaded %s weights: matched=%d missing=%d unexpected=%d skipped=%d mismatched=%d",
        source_name,
        len(filtered),
        len(msg.missing_keys),
        len(msg.unexpected_keys),
        len(skipped),
        len(mismatched),
    )
    if mismatched:
        _logger.warning("%s mismatched keys (first 10): %s", source_name, mismatched[:10])
    return msg


def _load_torch_file(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_openai_clip_vit_b16(model, checkpoint_path="checkpoints/ViT-B-16.pt", resize_pos_embed=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"OpenAI CLIP ViT-B/16 checkpoint not found: {checkpoint_path}")
    clip_model = torch.jit.load(checkpoint_path, map_location="cpu")
    raw_state = clip_model.state_dict()
    del clip_model

    converted = OrderedDict()
    for key, value in raw_state.items():
        if not key.startswith("visual."):
            continue
        key = key[len("visual."):]
        if key == "class_embedding":
            converted["cls_token"] = value.reshape(1, 1, -1)
        elif key == "positional_embedding":
            converted["pos_embed"] = _resize_pos_embed_if_needed(value, model, resize_pos_embed)
        elif key == "conv1.weight":
            converted["patch_embed.proj.weight"] = value
        elif key == "ln_pre.weight":
            converted["norm_pre.weight"] = value
        elif key == "ln_pre.bias":
            converted["norm_pre.bias"] = value
        elif key == "ln_post.weight":
            converted["norm.weight"] = value
        elif key == "ln_post.bias":
            converted["norm.bias"] = value
        elif key == "proj":
            continue
        elif key.startswith("transformer.resblocks."):
            parts = key.split(".")
            block_idx = parts[2]
            suffix = ".".join(parts[3:])
            prefix = f"blocks.{block_idx}."
            suffix = suffix.replace("ln_1", "norm1")
            suffix = suffix.replace("ln_2", "norm2")
            suffix = suffix.replace("attn.in_proj_weight", "attn.qkv.weight")
            suffix = suffix.replace("attn.in_proj_bias", "attn.qkv.bias")
            suffix = suffix.replace("attn.out_proj", "attn.proj")
            suffix = suffix.replace("mlp.c_fc", "mlp.fc1")
            suffix = suffix.replace("mlp.c_proj", "mlp.fc2")
            converted[prefix + suffix] = value
    return _load_filtered(model, converted, checkpoint_path)


def _convert_mae_qkv(state_dict):
    converted = OrderedDict()
    used = set()
    for key, value in state_dict.items():
        if ".attn.q_proj." in key or ".attn.k_proj." in key or ".attn.v_proj." in key:
            continue
        if key.startswith("decoder_") or key in {"mask_token"} or key.startswith("head."):
            continue
        if ".fc1." in key or ".fc2." in key:
            key = key.replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2.")
        converted[key] = value
    for idx in range(12):
        for suffix in ["weight", "bias"]:
            q_key = f"blocks.{idx}.attn.q_proj.{suffix}"
            k_key = f"blocks.{idx}.attn.k_proj.{suffix}"
            v_key = f"blocks.{idx}.attn.v_proj.{suffix}"
            out_key = f"blocks.{idx}.attn.qkv.{suffix}"
            if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                converted[out_key] = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
                used.update([q_key, k_key, v_key])
    return converted


def load_mae_vit_b16(model, checkpoint_path="checkpoints/mae_pretrain_vit_b.pth", resize_pos_embed=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MAE ViT-B/16 checkpoint not found: {checkpoint_path}")
    obj = _load_torch_file(checkpoint_path)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    state = _convert_mae_qkv(state)
    if "pos_embed" in state:
        state["pos_embed"] = _resize_pos_embed_if_needed(state["pos_embed"], model, resize_pos_embed)
    return _load_filtered(model, state, checkpoint_path)


def load_dinov2_vit_b14(model, checkpoint_path="checkpoints/dinov2_vitb14_pretrain.pth", resize_pos_embed=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DINOv2 ViT-B/14 checkpoint not found: {checkpoint_path}")
    state = _load_torch_file(checkpoint_path)
    state = state.get("model", state) if isinstance(state, dict) else state
    converted = OrderedDict()
    for key, value in state.items():
        if key == "mask_token" or key.startswith("register_tokens"):
            continue
        if key == "pos_embed":
            value = _resize_pos_embed_if_needed(value, model, resize_pos_embed)
        converted[key] = value
    return _load_filtered(model, converted, checkpoint_path)


def load_ibot21k_teacher(model, checkpoint_path="checkpoints/checkpoint.pth", resize_pos_embed=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"iBOT-21K checkpoint not found: {checkpoint_path}")
    obj = _load_torch_file(checkpoint_path)
    if not isinstance(obj, dict) or "teacher" not in obj:
        raise KeyError(f"Expected top-level 'teacher' in iBOT-21K checkpoint: {checkpoint_path}")
    state = OrderedDict()
    for key, value in obj["teacher"].items():
        key = key.replace("backbone.", "")
        if key == "pos_embed" and resize_pos_embed is not None:
            value = _resize_pos_embed_if_needed(value, model, resize_pos_embed)
        state[key] = value
    return _load_filtered(model, state, checkpoint_path)
