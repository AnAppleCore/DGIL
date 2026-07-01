# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

import logging
import os
from collections import OrderedDict
import torch
from backbone.pretrain_loaders import _load_torch_file



def _resize_pos_embed_adapter(pos_embed, model):
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)
    if tuple(pos_embed.shape) == tuple(model.pos_embed.shape):
        return pos_embed

    num_prefix_tokens = getattr(model, "num_tokens", 1)
    posemb_prefix = pos_embed[:, :num_prefix_tokens]
    posemb_grid = pos_embed[:, num_prefix_tokens:]
    old_size = int(math.sqrt(posemb_grid.shape[1]))
    new_grid_size = model.patch_embed.grid_size
    if isinstance(new_grid_size, tuple):
        new_h, new_w = new_grid_size
    else:
        new_h = new_w = int(new_grid_size)
    posemb_grid = posemb_grid.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(
        posemb_grid,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    return torch.cat([posemb_prefix, posemb_grid], dim=1)



def _filter_to_model(state_dict, model):
    model_state = model.state_dict()
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            filtered[key] = value.float() if torch.is_floating_point(value) else value
    return filtered



def _convert_standard_vit_to_adapter(state_dict):
    converted = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("head.") or key.startswith("decoder_") or key in {"mask_token"} or key.startswith("register_tokens"):
            continue
        if ".attn.qkv.weight" in key:
            q_weight, k_weight, v_weight = value.chunk(3, dim=0)
            converted[key.replace("qkv.weight", "q_proj.weight")] = q_weight
            converted[key.replace("qkv.weight", "k_proj.weight")] = k_weight
            converted[key.replace("qkv.weight", "v_proj.weight")] = v_weight
        elif ".attn.qkv.bias" in key:
            q_bias, k_bias, v_bias = value.chunk(3, dim=0)
            converted[key.replace("qkv.bias", "q_proj.bias")] = q_bias
            converted[key.replace("qkv.bias", "k_proj.bias")] = k_bias
            converted[key.replace("qkv.bias", "v_proj.bias")] = v_bias
        elif ".mlp.fc" in key:
            converted[key.replace(".mlp.", ".")] = value
        else:
            converted[key] = value
    return converted



def _load_and_freeze_adapter(model, state_dict):
    state_dict = _convert_standard_vit_to_adapter(state_dict)
    if "pos_embed" in state_dict:
        state_dict["pos_embed"] = _resize_pos_embed_adapter(state_dict["pos_embed"], model)
    filtered = _filter_to_model(state_dict, model)
    msg = model.load_state_dict(filtered, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    return model



def _load_ibot21k_adapter(model, checkpoint_path="checkpoints/checkpoint.pth"):
    obj = _load_torch_file(checkpoint_path)
    state = OrderedDict()
    for key, value in obj["teacher"].items():
        state[key.replace("backbone.", "")] = value
    return _load_and_freeze_adapter(model, state)



def _load_mae_adapter(model, checkpoint_path="checkpoints/mae_pretrain_vit_b.pth"):
    obj = _load_torch_file(checkpoint_path)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    return _load_and_freeze_adapter(model, state)



def _load_dinov2_adapter(model, checkpoint_path="checkpoints/dinov2_vitb14_pretrain.pth"):
    state = _load_torch_file(checkpoint_path)
    state = state.get("model", state) if isinstance(state, dict) else state
    return _load_and_freeze_adapter(model, state)



def _load_openai_clip_adapter(model, checkpoint_path="checkpoints/ViT-B-16.pt"):
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
            converted["pos_embed"] = value.reshape(1, value.shape[0], value.shape[1])
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
            suffix = suffix.replace("attn.out_proj", "attn.proj")
            suffix = suffix.replace("mlp.c_fc", "fc1")
            suffix = suffix.replace("mlp.c_proj", "fc2")
            out_key = prefix + suffix
            if suffix == "attn.in_proj_weight":
                q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                converted[prefix + "attn.q_proj.weight"] = q_weight
                converted[prefix + "attn.k_proj.weight"] = k_weight
                converted[prefix + "attn.v_proj.weight"] = v_weight
            elif suffix == "attn.in_proj_bias":
                q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                converted[prefix + "attn.q_proj.bias"] = q_bias
                converted[prefix + "attn.k_proj.bias"] = k_bias
                converted[prefix + "attn.v_proj.bias"] = v_bias
            else:
                converted[out_key] = value
    return _load_and_freeze_adapter(model, converted)



class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output





class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None,
                 init_values=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        if config.ffn_adapt:
            self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.mlp_drop(self.fc2(x))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                x = self.adaptmlp(x)
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + self.drop_path(self.ls2(x))
        return x





class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None, pre_norm=False):
        super().__init__()


        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, init_values=init_values,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x,)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def _create_vision_transformer(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
#     model = build_model_with_cfg(
#         VisionTransformer, variant, pretrained,
#         pretrained_cfg=pretrained_cfg,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)
#     return model




def vit_base_patch16_224_21k_ibot_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        model = _load_ibot21k_adapter(model)
    return model



def vit_base_patch16_224_clip_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), pre_norm=True, **kwargs)
    if pretrained:
        model = _load_openai_clip_adapter(model)
    return model



def vit_base_patch16_224_mae_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        model = _load_mae_adapter(model)
    return model



def vit_base_patch14_224_dinov2_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        init_values=1.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        model = _load_dinov2_adapter(model)
    return model



def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    from backbone import vit_dot_slca
    checkpoint_model = timm.create_model("vit_base_patch16_224_dot", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # s=model.state_dict()
    # # print the keys in s
    # for key in s.keys():
    #     print(key)
    # # print the keys in checkpoint_model
    # for key in state_dict.keys():
    #     if key in s.keys():
    #         print(key, 'yes')
    #     else:
    #         print(key, 'NOOOOOOOOOOOOOOOOOOO')

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model



def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    from backbone import vit_dot_slca
    checkpoint_model = timm.create_model("vit_base_patch16_224_dot", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # s=model.state_dict()
    # # print the keys in s
    # for key in s.keys():
    #     print(key)
    # # print the keys in checkpoint_model
    # for key in state_dict.keys():
    #     if key in s.keys():
    #         print(key, 'yes')
    #     else:
    #         print(key, 'NOOOOOOOOOOOOOOOOOOO')

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

