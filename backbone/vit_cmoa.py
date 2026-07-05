import torch
from torch import nn
from torch.nn import functional as F

from backbone import vit_adapter


class MixtureOfAdapters(nn.Module):
    def __init__(self, config, dropout=0.1):
        super().__init__()
        self.num_adapters = config.moa_num_adapters
        self.gamma = config.moa_cosine_gamma
        self.adapters = nn.ModuleList(
            [
                vit_adapter.Adapter(
                    config,
                    dropout=dropout,
                    bottleneck=config.ffn_num,
                    init_option=config.ffn_adapter_init_option,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                )
                for _ in range(self.num_adapters)
            ]
        )
        self.gate = nn.Linear(config.d_model, self.num_adapters)
        self.latest_cosine_loss = None

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        adapter_outputs = [adapter(x, add_residual=False) for adapter in self.adapters]
        stacked = torch.stack(adapter_outputs, dim=1)
        pooled = x.mean(dim=1)
        weights = torch.softmax(self.gate(pooled), dim=-1).view(x.size(0), self.num_adapters, 1, 1)
        mixed = torch.sum(stacked * weights, dim=1)
        self.latest_cosine_loss = self._cosine_diversity_loss(adapter_outputs)
        if add_residual:
            return mixed + residual
        return mixed

    def _cosine_diversity_loss(self, adapter_outputs):
        if len(adapter_outputs) <= 1:
            return adapter_outputs[0].new_tensor(0.0)
        losses = []
        for i in range(len(adapter_outputs)):
            for j in range(i + 1, len(adapter_outputs)):
                cosine = F.cosine_similarity(adapter_outputs[i], adapter_outputs[j], dim=-1)
                losses.append(torch.relu(cosine - self.gamma).mean())
        return torch.stack(losses).mean()


def _install_moa(model, tuning_config):
    for block in model.blocks:
        if hasattr(block, "adaptmlp"):
            block.adaptmlp = MixtureOfAdapters(tuning_config, dropout=tuning_config.moa_dropout)
    model.get_moa_loss = lambda: _collect_moa_loss(model)
    return model


def _collect_moa_loss(model):
    losses = []
    for block in model.blocks:
        module = getattr(block, "adaptmlp", None)
        loss = getattr(module, "latest_cosine_loss", None)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(losses).mean()


def vit_base_patch16_224_cmoa(pretrained=False, **kwargs):
    tuning_config = kwargs["tuning_config"]
    model = vit_adapter.vit_base_patch16_224_adapter(pretrained=pretrained, **kwargs)
    return _install_moa(model, tuning_config)


def vit_base_patch16_224_21k_ibot_cmoa(pretrained=False, **kwargs):
    tuning_config = kwargs["tuning_config"]
    model = vit_adapter.vit_base_patch16_224_21k_ibot_adapter(pretrained=pretrained, **kwargs)
    return _install_moa(model, tuning_config)


def vit_base_patch16_224_clip_cmoa(pretrained=False, **kwargs):
    tuning_config = kwargs["tuning_config"]
    model = vit_adapter.vit_base_patch16_224_clip_adapter(pretrained=pretrained, **kwargs)
    return _install_moa(model, tuning_config)


def vit_base_patch16_224_mae_cmoa(pretrained=False, **kwargs):
    tuning_config = kwargs["tuning_config"]
    model = vit_adapter.vit_base_patch16_224_mae_adapter(pretrained=pretrained, **kwargs)
    return _install_moa(model, tuning_config)


def vit_base_patch14_224_dinov2_cmoa(pretrained=False, **kwargs):
    tuning_config = kwargs["tuning_config"]
    model = vit_adapter.vit_base_patch14_224_dinov2_adapter(pretrained=pretrained, **kwargs)
    return _install_moa(model, tuning_config)
