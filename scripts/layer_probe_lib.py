import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_imports() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


DATASET_DEFAULTS = {
    "digitsdg": {"init_cls": 2, "increment": 2},
    "officehome": {"init_cls": 13, "increment": 13},
    "core50": {"init_cls": 5, "increment": 5},
    "domainnet": {"init_cls": 35, "increment": 35},
}

BACKBONE_ALIASES = {
    "default": "vit_base_patch16_224_dot",
    "ibot": "vit_base_patch16_224_21k_ibot_dot",
    "ibot_21k": "vit_base_patch16_224_21k_ibot_dot",
    "mae": "vit_base_patch16_224_mae_dot",
    "dinov2": "vit_base_patch14_224_dinov2_dot",
    "clip": "vit_base_patch16_224_clip_dot",
}

SLCA_BACKBONE_ALIASES = {
    "default": "vit_base_patch16_224",
    "ibot": "vit_base_patch16_224_21k_ibot",
    "ibot_21k": "vit_base_patch16_224_21k_ibot",
    "mae": "vit_base_patch16_224_mae",
    "dinov2": "vit_base_patch14_224_dinov2",
    "clip": "vit_base_patch16_224_clip",
}


def canonical_backbone(name: str, feature_source: str = "raw_pretrained") -> str:
    if feature_source == "slca_trained":
        return SLCA_BACKBONE_ALIASES.get(name, name)
    return BACKBONE_ALIASES.get(name, name)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


DEFAULT_RUN_NAMES = {
    "raw_pretrained": "default",
    "slca_trained": "seed1994_final",
}


def source_output_dir(results_root: Path, feature_source: str) -> Path:
    if feature_source not in DEFAULT_RUN_NAMES:
        raise ValueError(f"Unknown feature source: {feature_source}")
    return Path(results_root) / feature_source


def canonical_run_name(feature_source: str, run_name: str | None = None) -> str:
    if feature_source not in DEFAULT_RUN_NAMES:
        raise ValueError(f"Unknown feature source: {feature_source}")
    return safe_name(run_name or DEFAULT_RUN_NAMES[feature_source])


def analysis_result_dir(
    source_dir: Path,
    analysis: str,
    dataset: str,
    backbone: str,
    run_name: str,
    normalization: str,
    method: str | None = None,
    protocol: str | None = None,
) -> Path:
    path = source_dir / "analyses" / analysis
    if method:
        path = path / safe_name(method)
    path = path / dataset / safe_name(backbone) / safe_name(run_name) / normalization
    if protocol:
        path = path / safe_name(protocol)
    return path


def figure_output_dir(source_dir: Path, analysis: str, method: str | None = None) -> Path:
    path = source_dir / "figures" / analysis
    return path / safe_name(method) if method else path


def backbone_required_checkpoints(backbone: str) -> List[str]:
    if backbone in {"vit_base_patch16_224_dot", "pretrained_vit_b16_224_dot", "vit_base_patch16_224"}:
        return ["B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"]
    if backbone in {"vit_base_patch16_224_21k_ibot_dot", "vit_base_patch16_224_21k_ibot"}:
        return ["checkpoint.pth"]
    if backbone in {"vit_base_patch16_224_mae_dot", "vit_base_patch16_224_mae"}:
        return ["mae_pretrain_vit_b.pth"]
    if backbone in {"vit_base_patch14_224_dinov2_dot", "vit_base_patch14_224_dinov2"}:
        return ["dinov2_vitb14_pretrain.pth"]
    if backbone in {"vit_base_patch16_224_clip_dot", "vit_base_patch16_224_clip"}:
        return ["ViT-B-16.pt"]
    return []


def check_required_checkpoints(backbone: str, checkpoint_dir: Path) -> Tuple[bool, List[str]]:
    missing = []
    for rel in backbone_required_checkpoints(backbone):
        if not (checkpoint_dir / rel).exists():
            missing.append(str(checkpoint_dir / rel))
    return len(missing) == 0, missing


def build_probe_args(dataset: str, backbone: str, batch_size: int, device: str, model_name: str = "dot_slca") -> dict:
    defaults = DATASET_DEFAULTS[dataset]
    return {
        "prefix": "layer_probe",
        "dataset": dataset,
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "shuffle": False,
        "init_cls": defaults["init_cls"],
        "increment": defaults["increment"],
        "model_name": model_name,
        "model_postfix": "layer_probe",
        "backbone_type": backbone,
        "device": [device],
        "seed": 1994,
        "batch_size": batch_size,
        "enable_dgil": False,
        "random_reference": False,
        "reference_domain_id": 0,
        "multi_domain_base_task": False,
        "pretrained": True,
    }


def apply_input_normalization(data_manager, normalization: str) -> None:
    if normalization == "repo":
        return
    ensure_project_imports()
    from torchvision import transforms

    if normalization == "imagenet":
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    elif normalization == "clip":
        norm = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    data_manager._common_trsf = [*data_manager._common_trsf, norm]


def ensure_original_train_aliases(data_manager) -> None:
    if not hasattr(data_manager, "_original_train_data"):
        data_manager._original_train_data = data_manager._train_data
    if not hasattr(data_manager, "_original_train_targets"):
        data_manager._original_train_targets = data_manager._train_targets
    if not hasattr(data_manager, "_original_train_domain_idx"):
        if hasattr(data_manager, "original_train_domain_idx"):
            data_manager._original_train_domain_idx = data_manager.original_train_domain_idx
        else:
            raise AttributeError("DomainDataManager lacks original train domain indices")


def build_data_manager(dataset: str, args: dict, normalization: str):
    ensure_project_imports()
    from utils.domain_data_manager import DomainDataManager

    defaults = DATASET_DEFAULTS[dataset]
    dm = DomainDataManager(dataset, False, args["seed"], defaults["init_cls"], defaults["increment"], args)
    ensure_original_train_aliases(dm)
    apply_input_normalization(dm, normalization)
    return dm


def build_backbone(args: dict):
    ensure_project_imports()
    from utils.inc_net import get_backbone

    model = get_backbone(args, pretrained=True)
    model.eval()
    return model


def load_slca_checkpoint_into_backbone(model, checkpoint_path: Path) -> dict:
    import torch

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model_state = model.state_dict()
    filtered = {}
    mismatched = []
    skipped = []
    for raw_key, value in state_dict.items():
        key = raw_key
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        elif key.startswith("_network.backbone."):
            key = key[len("_network.backbone."):]
        else:
            skipped.append(raw_key)
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            mismatched.append(raw_key)
    if not filtered:
        raise RuntimeError(f"No backbone weights loaded from {checkpoint_path}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    required_prefixes = ["patch_embed", "pos_embed", "blocks.0", "blocks.11"]
    loaded_keys = set(filtered)
    missing_required = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in loaded_keys)]
    if missing_required:
        raise RuntimeError(f"Checkpoint {checkpoint_path} missing required backbone prefixes: {missing_required}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_keys": len(filtered),
        "skipped_non_backbone_keys": len(skipped),
        "mismatched_keys": mismatched[:50],
        "num_mismatched_keys": len(mismatched),
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
        "checkpoint_tasks": payload.get("tasks") if isinstance(payload, dict) else None,
        "checkpoint_dataset": payload.get("dataset") if isinstance(payload, dict) else None,
        "checkpoint_backbone_type": payload.get("backbone_type") if isinstance(payload, dict) else None,
        "checkpoint_seed": payload.get("seed") if isinstance(payload, dict) else None,
    }


def layerwise_forward(model, inputs):
    import torch

    with torch.no_grad():
        x = model.patch_embed(inputs)
        x = model._pos_embed(x)
        x = model.norm_pre(x)
        cls_features = []
        mean_features = []
        num_prefix = int(getattr(model, "num_prefix_tokens", 1) or 0)
        for block in model.blocks:
            x = block(x)
            if num_prefix > 0:
                cls_features.append(x[:, 0])
                mean_features.append(x[:, num_prefix:].mean(dim=1))
            else:
                cls_features.append(x.mean(dim=1))
                mean_features.append(x.mean(dim=1))
        cls = torch.stack(cls_features, dim=1).detach().cpu().float().numpy()
        mean = torch.stack(mean_features, dim=1).detach().cpu().float().numpy()
    return cls, mean


def balanced_indices(labels: np.ndarray, max_per_class: Optional[int], seed: int) -> np.ndarray:
    if max_per_class is None or max_per_class <= 0:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(np.sort(idx))
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=np.int64)


def feature_cache_dir(output_dir: Path, dataset: str, backbone: str, normalization: str, run_name: str) -> Path:
    return output_dir / "features" / dataset / safe_name(backbone) / safe_name(run_name) / normalization


def feature_shard_path(output_dir: Path, dataset: str, backbone: str, normalization: str, split: str, domain_id: int, run_name: str) -> Path:
    return feature_cache_dir(output_dir, dataset, backbone, normalization, run_name) / f"{split}_domain{domain_id:02d}.npz"


def load_feature_split(output_dir: Path, dataset: str, backbone: str, normalization: str, split: str, run_name: str):
    cache_dir = feature_cache_dir(output_dir, dataset, backbone, normalization, run_name)
    shard_paths = sorted(cache_dir.glob(f"{split}_domain*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found for {dataset}/{backbone}/{normalization}/{split} in {cache_dir}")
    cls, mean, y_class, y_domain, sample_indices = [], [], [], [], []
    metadata = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            cls.append(data["features_cls"].astype(np.float32, copy=False))
            mean.append(data["features_mean"].astype(np.float32, copy=False))
            y_class.append(data["class_ids"].astype(np.int64, copy=False))
            y_domain.append(data["domain_ids"].astype(np.int64, copy=False))
            sample_indices.append(data["sample_indices"].astype(np.int64, copy=False))
            if "metadata_json" in data:
                metadata.append(json.loads(str(data["metadata_json"])))
    return {
        "cls": np.concatenate(cls, axis=0),
        "mean": np.concatenate(mean, axis=0),
        "class": np.concatenate(y_class, axis=0),
        "domain": np.concatenate(y_domain, axis=0),
        "sample_indices": np.concatenate(sample_indices, axis=0),
        "metadata": metadata,
        "shards": [str(p) for p in shard_paths],
    }


def load_feature_layer(
    output_dir: Path,
    dataset: str,
    backbone: str,
    normalization: str,
    split: str,
    feature_pool: str,
    layer: int,
    run_name: str,
    domain_ids: Optional[Sequence[int]] = None,
) -> dict:
    if feature_pool not in {"cls", "mean"}:
        raise ValueError(f"Unknown feature pool: {feature_pool}")
    if layer < 1:
        raise ValueError(f"Layer must be one-indexed and positive, got {layer}")
    cache_dir = feature_cache_dir(output_dir, dataset, backbone, normalization, run_name)
    requested_domains = None if domain_ids is None else {int(value) for value in domain_ids}
    shard_paths = sorted(cache_dir.glob(f"{split}_domain*.npz"))
    if requested_domains is not None:
        shard_paths = [
            path for path in shard_paths
            if int(path.stem.rsplit("domain", 1)[1]) in requested_domains
        ]
    if not shard_paths:
        raise FileNotFoundError(
            f"No feature shards found for {dataset}/{backbone}/{normalization}/{split} in {cache_dir}"
        )

    feature_key = f"features_{feature_pool}"
    features, y_class, y_domain, sample_indices = [], [], [], []
    metadata = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            shard_features = data[feature_key]
            if shard_features.ndim != 3 or layer > shard_features.shape[1]:
                raise RuntimeError(
                    f"Expected {feature_key} shaped [N,L,D] with L >= {layer}, got {shard_features.shape} in {path}"
                )
            features.append(shard_features[:, layer - 1, :].astype(np.float32, copy=False))
            y_class.append(data["class_ids"].astype(np.int64, copy=False))
            y_domain.append(data["domain_ids"].astype(np.int64, copy=False))
            sample_indices.append(data["sample_indices"].astype(np.int64, copy=False))
            if "metadata_json" in data:
                metadata.append(json.loads(str(data["metadata_json"])))
    return {
        "features": np.concatenate(features, axis=0),
        "class": np.concatenate(y_class, axis=0),
        "domain": np.concatenate(y_domain, axis=0),
        "sample_indices": np.concatenate(sample_indices, axis=0),
        "metadata": metadata,
        "shards": [str(path) for path in shard_paths],
        "feature_pool": feature_pool,
        "layer": int(layer),
    }


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def apply_preprocess(x_train: np.ndarray, x_test: np.ndarray, mode: str):
    if mode == "none":
        return x_train, x_test, {}
    if mode == "l2":
        return l2_normalize(x_train), l2_normalize(x_test), {"preprocess": "l2"}
    if mode == "standard":
        mean, std = standardize_fit(x_train)
        return (x_train - mean) / std, (x_test - mean) / std, {"preprocess": "standard"}
    raise ValueError(f"Unknown preprocess mode: {mode}")


def compute_prototypes(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(y)
    means = []
    counts = []
    for label in labels:
        idx = y == label
        means.append(x[idx].mean(axis=0))
        counts.append(int(idx.sum()))
    return labels.astype(np.int64), np.stack(means).astype(np.float32), np.array(counts, dtype=np.int64)


def predict_nearest_mean(x: np.ndarray, labels: np.ndarray, means: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    preds = []
    mean_sq = np.sum(means * means, axis=1, keepdims=True).T
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        dists = np.sum(xb * xb, axis=1, keepdims=True) - 2.0 * xb @ means.T + mean_sq
        preds.append(labels[np.argmin(dists, axis=1)])
    return np.concatenate(preds)


def ncm_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, normalize: bool = True):
    if normalize:
        x_train = l2_normalize(x_train)
        x_test = l2_normalize(x_test)
    labels, means, counts = compute_prototypes(x_train, y_train)
    if normalize:
        means = l2_normalize(means)
    return predict_nearest_mean(x_test, labels, means), {"prototype_counts": counts.tolist(), "normalize": normalize}


@dataclass
class WhitenedNCMModel:
    labels: np.ndarray
    means_whitened: np.ndarray
    whitener: np.ndarray
    normalize: bool
    metadata: dict


def _allocate_stratified_sample_counts(group_sizes: np.ndarray, max_samples: int) -> np.ndarray:
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    if max_samples <= 0 or int(group_sizes.sum()) <= max_samples:
        return group_sizes.copy()
    counts = np.zeros_like(group_sizes)
    active = np.flatnonzero(group_sizes > 0)
    remaining = int(max_samples)
    while remaining > 0 and len(active):
        share = max(remaining // len(active), 1)
        room = group_sizes[active] - counts[active]
        additions = np.minimum(room, share)
        counts[active] += additions
        used = int(additions.sum())
        remaining -= used
        active = active[counts[active] < group_sizes[active]]
        if used == 0:
            break
    return counts


def stratified_covariance_indices(
    targets: np.ndarray,
    cross_factors: Optional[np.ndarray],
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, dict]:
    targets = np.asarray(targets)
    if cross_factors is None:
        cross_factors = np.zeros(len(targets), dtype=np.int64)
    cross_factors = np.asarray(cross_factors)
    if len(targets) != len(cross_factors):
        raise ValueError("targets and cross_factors must have identical lengths")

    groups = {}
    for idx, (target, factor) in enumerate(zip(targets.tolist(), cross_factors.tolist())):
        groups.setdefault((int(target), int(factor)), []).append(idx)
    ordered_groups = sorted(groups)
    group_sizes = np.array([len(groups[key]) for key in ordered_groups], dtype=np.int64)
    sample_counts = _allocate_stratified_sample_counts(group_sizes, max_samples)
    rng = np.random.default_rng(seed)
    sampled = []
    group_metadata = []
    for key, available, requested in zip(ordered_groups, group_sizes.tolist(), sample_counts.tolist()):
        candidates = np.asarray(groups[key], dtype=np.int64)
        if requested < available:
            chosen = np.sort(rng.choice(candidates, size=requested, replace=False))
        else:
            chosen = candidates
        sampled.append(chosen)
        group_metadata.append({
            "target": int(key[0]),
            "cross_factor": int(key[1]),
            "available": int(available),
            "sampled": int(len(chosen)),
        })
    indices = np.sort(np.concatenate(sampled)) if sampled else np.array([], dtype=np.int64)
    metadata = {
        "strategy": "target_then_cross_factor_balanced",
        "requested_max_samples": int(max_samples),
        "available_samples": int(len(targets)),
        "sampled_samples": int(len(indices)),
        "groups": group_metadata,
    }
    return indices, metadata


def fit_whitened_ncm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    normalize: bool = True,
    reg: float = 1e-4,
    cov_max_samples: int = 200000,
    seed: int = 1994,
    covariance_cross_factors: Optional[np.ndarray] = None,
    progress=None,
) -> WhitenedNCMModel:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have identical lengths")
    if normalize:
        if progress:
            progress("wncm normalize start")
        x_train = l2_normalize(x_train)
        if progress:
            progress("wncm normalize done")
    if progress:
        progress("wncm prototypes start")
    labels, means, counts = compute_prototypes(x_train, y_train)
    if progress:
        progress(f"wncm prototypes done n_labels={len(labels)}")
    label_to_pos = {int(label): idx for idx, label in enumerate(labels.tolist())}
    prototype_positions = np.fromiter((label_to_pos[int(label)] for label in y_train), dtype=np.int64, count=len(y_train))
    sample_indices, sampling_meta = stratified_covariance_indices(
        y_train,
        covariance_cross_factors,
        cov_max_samples,
        seed,
    )
    if not len(sample_indices):
        raise ValueError("Cannot fit WNCM with an empty covariance sample")
    if progress:
        progress(
            f"wncm residuals start total={len(x_train)} cov_samples={len(sample_indices)} dim={x_train.shape[1]}"
        )
    residuals_cov = x_train[sample_indices] - means[prototype_positions[sample_indices]]
    if progress:
        progress("wncm covariance start")
    cov = (residuals_cov.T @ residuals_cov) / max(len(residuals_cov) - 1, 1)
    avg_var = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    cov = cov + (reg * max(avg_var, 1e-12)) * np.eye(cov.shape[0], dtype=cov.dtype)
    if progress:
        progress("wncm covariance done; eigh start")
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    eigvals = np.maximum(eigvals, 1e-12)
    whitener = ((eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T).astype(np.float32)
    means_whitened = (means @ whitener).astype(np.float32)
    if progress:
        progress("wncm fit done")
    metadata = {
        "prototype_labels": labels.tolist(),
        "prototype_counts": counts.tolist(),
        "normalize": bool(normalize),
        "reg": float(reg),
        "cov_samples": int(len(sample_indices)),
        "covariance_sampling": sampling_meta,
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
    }
    return WhitenedNCMModel(labels, means_whitened, whitener, normalize, metadata)


def predict_whitened_ncm(model: WhitenedNCMModel, x_test: np.ndarray, progress=None) -> np.ndarray:
    x_test = np.asarray(x_test, dtype=np.float32)
    if model.normalize:
        x_test = l2_normalize(x_test)
    if progress:
        progress("wncm projection start")
    x_test_whitened = (x_test @ model.whitener).astype(np.float32)
    if progress:
        progress("wncm nearest-mean predict start")
    pred = predict_nearest_mean(x_test_whitened, model.labels, model.means_whitened)
    if progress:
        progress("wncm nearest-mean predict done")
    return pred


def whitened_ncm_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    normalize: bool = True,
    reg: float = 1e-4,
    cov_max_samples: int = 200000,
    seed: int = 1994,
    covariance_cross_factors: Optional[np.ndarray] = None,
    progress=None,
):
    model = fit_whitened_ncm(
        x_train,
        y_train,
        normalize=normalize,
        reg=reg,
        cov_max_samples=cov_max_samples,
        seed=seed,
        covariance_cross_factors=covariance_cross_factors,
        progress=progress,
    )
    pred = predict_whitened_ncm(model, x_test, progress=progress)
    return pred, model.metadata


def accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: Optional[Dict[int, str]] = None) -> dict:
    labels = np.unique(y_true)
    total = float((y_true == y_pred).mean() * 100.0) if len(y_true) else math.nan
    per_label = {}
    recalls = []
    for label in labels:
        idx = y_true == label
        acc = float((y_true[idx] == y_pred[idx]).mean() * 100.0) if idx.any() else math.nan
        key = str(label_names.get(int(label), int(label))) if label_names else str(int(label))
        per_label[key] = acc
        if not math.isnan(acc):
            recalls.append(acc)
    return {
        "accuracy": total,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else math.nan,
        "per_label_accuracy": per_label,
        "num_samples": int(len(y_true)),
    }


def class_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, init_cls: int, increment: int) -> dict:
    out = {}
    max_label = int(np.max(y_true)) if len(y_true) else -1
    ranges = [(0, init_cls)]
    start = init_cls
    while start <= max_label:
        ranges.append((start, min(start + increment, max_label + 1)))
        start += increment
    for lo, hi in ranges:
        idx = (y_true >= lo) & (y_true < hi)
        if idx.any():
            out[f"{lo:02d}-{hi - 1:02d}"] = float((y_true[idx] == y_pred[idx]).mean() * 100.0)
    return out


def stratified_train_val_indices(y: np.ndarray, val_fraction: float, seed: int, min_per_class: int = 2):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        if len(idx) < min_per_class:
            train.append(idx)
            continue
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 1)
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return np.concatenate(train), np.concatenate(val) if val else np.array([], dtype=np.int64)


def fit_predict_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    dataset: str,
    target: str,
    method: str = "auto",
    seed: int = 1994,
    progress=None,
):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    import warnings

    n_samples = len(x_train)
    n_classes = len(np.unique(y_train))
    if method == "auto":
        if dataset == "domainnet" and target == "class":
            method = "ridge"
        elif n_samples > 120000 or n_classes > 150:
            method = "ridge"
        else:
            method = "logreg"

    meta = {"linear_method": method, "n_train": int(n_samples), "n_classes": int(n_classes)}
    if method == "ridge":
        alphas = [0.1, 1.0, 10.0, 100.0]
        if progress:
            progress(f"linear ridge split start n_train={n_samples} n_classes={n_classes}")
        tr_idx, val_idx = stratified_train_val_indices(y_train, 0.2, seed)
        if progress:
            progress(f"linear ridge split done train_subset={len(tr_idx)} val={len(val_idx)}")
        best_alpha, best_acc = alphas[0], -1.0
        for alpha in alphas:
            if progress:
                progress(f"linear ridge alpha={alpha} fit start")
            clf = RidgeClassifier(alpha=alpha, class_weight="balanced" if target == "class" else None, solver="lsqr")
            clf.fit(x_train[tr_idx], y_train[tr_idx])
            if progress:
                progress(f"linear ridge alpha={alpha} fit done; val predict start")
            if len(val_idx):
                pred = clf.predict(x_train[val_idx])
                acc = float((pred == y_train[val_idx]).mean())
            else:
                acc = 0.0
            if progress:
                progress(f"linear ridge alpha={alpha} val done acc={acc * 100.0:.4f}")
            if acc > best_acc:
                best_acc, best_alpha = acc, alpha
        if progress:
            progress(f"linear ridge final fit start best_alpha={best_alpha}")
        clf = RidgeClassifier(alpha=best_alpha, class_weight="balanced" if target == "class" else None, solver="lsqr")
        clf.fit(x_train, y_train)
        if progress:
            progress("linear ridge final fit done; test predict start")
        pred = clf.predict(x_test)
        if progress:
            progress("linear ridge test predict done")
        meta.update({"alpha": best_alpha, "val_accuracy": best_acc * 100.0, "class_weight": clf.class_weight, "solver": "lsqr"})
        return pred, meta

    if method == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced" if target == "class" else None,
            random_state=seed,
        )
        clf.fit(x_train, y_train)
        meta.update({"alpha": clf.alpha, "max_iter": clf.max_iter, "class_weight": clf.class_weight})
        return clf.predict(x_test), meta

    if method == "logreg":
        cs = [0.01, 0.1, 1.0, 10.0]
        tr_idx, val_idx = stratified_train_val_indices(y_train, 0.2, seed)
        best_c, best_acc, best_iter = cs[0], -1.0, None
        for c in cs:
            clf = LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced" if target == "class" else None,
                n_jobs=1,
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(x_train[tr_idx], y_train[tr_idx])
            pred = clf.predict(x_train[val_idx]) if len(val_idx) else clf.predict(x_train[tr_idx])
            truth = y_train[val_idx] if len(val_idx) else y_train[tr_idx]
            acc = float((pred == truth).mean())
            if acc > best_acc:
                best_acc, best_c = acc, c
                best_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else None
        clf = LogisticRegression(
            C=best_c,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced" if target == "class" else None,
            n_jobs=1,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(x_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        meta.update({
            "C": best_c,
            "val_accuracy": best_acc * 100.0,
            "solver": "lbfgs",
            "max_iter": clf.max_iter,
            "n_iter": int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else best_iter,
            "class_weight": clf.class_weight,
            "converged": converged,
        })
        return clf.predict(x_test), meta

    raise ValueError(f"Unknown linear method: {method}")
