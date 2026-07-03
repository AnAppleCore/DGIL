import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from layer_probe_lib import (
    DATASET_DEFAULTS,
    atomic_write_json,
    balanced_indices,
    build_backbone,
    build_data_manager,
    build_probe_args,
    canonical_backbone,
    check_required_checkpoints,
    feature_cache_dir,
    feature_shard_path,
    layerwise_forward,
    load_slca_checkpoint_into_backbone,
    project_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract layer-wise pretrained ViT features for class/domain probing.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    parser.add_argument("--backbone", required=True, help="Backbone name or alias: default, ibot, mae, dinov2, clip.")
    parser.add_argument("--output-dir", default="results/layer_probe")
    parser.add_argument("--normalization", default="repo", choices=["repo", "imagenet", "clip"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--domains", nargs="+", default=None, help="Optional domain ids to extract.")
    parser.add_argument("--max-samples-per-class", type=int, default=0, help="Pilot-only balanced cap per semantic class within each domain shard.")
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--feature-source", default="raw_pretrained", choices=["raw_pretrained", "slca_trained"])
    parser.add_argument("--checkpoint-path", default=None, help="Required when --feature-source=slca_trained.")
    parser.add_argument("--run-name", default=None, help="Optional run namespace under the backbone feature cache.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_shard(path: Path, features_cls, features_mean, class_ids, domain_ids, sample_indices, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(
        tmp,
        features_cls=features_cls.astype(np.float32, copy=False),
        features_mean=features_mean.astype(np.float32, copy=False),
        class_ids=class_ids.astype(np.int64, copy=False),
        domain_ids=domain_ids.astype(np.int64, copy=False),
        sample_indices=sample_indices.astype(np.int64, copy=False),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    generated = Path(str(tmp) + ".npz")
    if generated.exists():
        generated.replace(path)
    else:
        tmp.replace(path)


def extract_domain_split(model, data_manager, args, split: str, domain_id: int, device: torch.device):
    indices = np.arange(data_manager.nb_classes)
    _, targets, dataset = data_manager.get_domain_dataset(indices, source=split, mode="test", domain_id=domain_id, ret_data=True)
    selected = balanced_indices(targets, args.max_samples_per_class if args.max_samples_per_class > 0 else None, args.seed)
    if len(selected) != len(targets):
        data = dataset.images[selected]
        labels = np.asarray(dataset.labels)[selected]
        from utils.data_manager import DummyDataset
        dataset = DummyDataset(data, labels, dataset.trsf, dataset.use_path)
        targets = labels
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cls_parts, mean_parts, class_parts, domain_parts, sample_parts = [], [], [], [], []
    offset = 0
    for _, inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        cls, mean = layerwise_forward(model, inputs)
        labels_np = labels.numpy().astype(np.int64)
        batch_n = len(labels_np)
        cls_parts.append(cls)
        mean_parts.append(mean)
        class_parts.append(labels_np)
        domain_parts.append(np.full(batch_n, domain_id, dtype=np.int64))
        sample_parts.append(np.arange(offset, offset + batch_n, dtype=np.int64))
        offset += batch_n
    if not cls_parts:
        raise RuntimeError(f"No samples for split={split} domain_id={domain_id}")
    return (
        np.concatenate(cls_parts, axis=0),
        np.concatenate(mean_parts, axis=0),
        np.concatenate(class_parts, axis=0),
        np.concatenate(domain_parts, axis=0),
        np.concatenate(sample_parts, axis=0),
    )


def main():
    args = parse_args()
    root = project_root()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    backbone = canonical_backbone(args.backbone, args.feature_source)
    checkpoint_ok, missing = check_required_checkpoints(backbone, root / "checkpoints")
    if not checkpoint_ok:
        raise FileNotFoundError("Missing required checkpoint files: " + ", ".join(missing))
    if args.feature_source == "slca_trained" and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when --feature-source=slca_trained")

    os.environ.setdefault("DGIL_DATA_ROOT", "/data/datasets")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = "slca" if args.feature_source == "slca_trained" else "dot_slca"
    probe_args = build_probe_args(args.dataset, backbone, args.batch_size, "0", model_name=model_name)
    data_manager = build_data_manager(args.dataset, probe_args, args.normalization)
    domains = [int(d) for d in args.domains] if args.domains else list(range(data_manager.num_domains))

    model = build_backbone(probe_args)
    checkpoint_meta = None
    if args.feature_source == "slca_trained":
        checkpoint_meta = load_slca_checkpoint_into_backbone(model, Path(args.checkpoint_path))
        print(f"[checkpoint-loaded] {checkpoint_meta}", flush=True)
    model = model.to(torch_device)
    model.eval()

    cache_dir = feature_cache_dir(output_dir, args.dataset, backbone, args.normalization, args.run_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "backbone": backbone,
        "normalization": args.normalization,
        "feature_source": args.feature_source,
        "run_name": args.run_name,
        "checkpoint_meta": checkpoint_meta,
        "splits": args.splits,
        "domains": domains,
        "num_classes": int(data_manager.nb_classes),
        "num_domains": int(data_manager.num_domains),
        "domain_names": list(data_manager.domain_names),
        "batch_size": args.batch_size,
        "max_samples_per_class": args.max_samples_per_class,
        "shards": [],
    }

    for split in args.splits:
        for domain_id in domains:
            shard = feature_shard_path(output_dir, args.dataset, backbone, args.normalization, split, domain_id, args.run_name)
            if shard.exists() and not args.overwrite:
                print(f"[skip] existing shard {shard}", flush=True)
                summary["shards"].append(str(shard))
                continue
            print(f"[extract] dataset={args.dataset} backbone={backbone} norm={args.normalization} split={split} domain={domain_id}", flush=True)
            features_cls, features_mean, class_ids, domain_ids, sample_indices = extract_domain_split(
                model, data_manager, args, split, domain_id, torch_device
            )
            if features_cls.ndim != 3 or features_cls.shape[1] != 12:
                raise RuntimeError(f"Expected cls feature shape [N,12,D], got {features_cls.shape}")
            if features_mean.shape != features_cls.shape:
                raise RuntimeError(f"Mean feature shape mismatch: cls={features_cls.shape}, mean={features_mean.shape}")
            metadata = {
                "dataset": args.dataset,
                "backbone": backbone,
                "normalization": args.normalization,
                "split": split,
                "domain_id": domain_id,
                "domain_name": data_manager.domain_names[domain_id],
                "num_samples": int(len(class_ids)),
                "feature_shape": list(features_cls.shape),
                "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(class_ids, return_counts=True))},
                "domain_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(domain_ids, return_counts=True))},
                "pooling": ["cls", "mean"],
            }
            save_shard(shard, features_cls, features_mean, class_ids, domain_ids, sample_indices, metadata)
            print(f"[saved] {shard} shape={features_cls.shape}", flush=True)
            summary["shards"].append(str(shard))

    atomic_write_json(cache_dir / "summary.json", summary)
    print(f"[done] summary={cache_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
