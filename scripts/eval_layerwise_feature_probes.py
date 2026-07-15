import argparse
import time
from pathlib import Path

import numpy as np

from layer_probe_lib import (
    DATASET_DEFAULTS,
    accuracy_metrics,
    append_csv,
    analysis_result_dir,
    atomic_write_json,
    canonical_backbone,
    canonical_run_name,
    class_group_accuracy,
    feature_cache_dir,
    fit_predict_linear,
    load_feature_split,
    ncm_predict,
    project_root,
    standardize_fit,
    whitened_ncm_predict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate layer-wise feature probes from cached features.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    parser.add_argument("--backbone", required=True, help="Backbone name or alias: default, ibot, mae, dinov2, clip.")
    parser.add_argument("--output-dir", default="results/layer_probe/raw_pretrained")
    parser.add_argument("--normalization", default="repo", choices=["repo", "imagenet", "clip"])
    parser.add_argument("--feature-pools", nargs="+", default=["cls", "mean"], choices=["cls", "mean"])
    parser.add_argument("--targets", nargs="+", default=["class", "domain"], choices=["class", "domain"])
    parser.add_argument("--classifiers", nargs="+", default=["linear", "ncm", "wncm"], choices=["linear", "ncm", "wncm"])
    parser.add_argument("--linear-method", default="auto", choices=["auto", "logreg", "ridge", "sgd"])
    parser.add_argument("--feature-source", default="raw_pretrained", choices=["raw_pretrained", "slca_trained"])
    parser.add_argument("--run-name", default=None, help="Optional run namespace under the backbone feature cache/result tree.")
    parser.add_argument("--preprocess", default="standard", choices=["none", "l2", "standard"])
    parser.add_argument("--wncm-reg", type=float, default=1e-4)
    parser.add_argument("--cov-max-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-progress", action="store_true", help="Print step-level progress before long computations.")
    return parser.parse_args()


def preprocess_for_classifier(x_train, x_test, classifier, preprocess):
    from layer_probe_lib import apply_preprocess, l2_normalize

    if classifier in {"ncm", "wncm"}:
        if preprocess == "standard":
            train, test, meta = apply_preprocess(x_train, x_test, "standard")
            return train, test, {**meta, "ncm_internal_l2": True}
        return x_train, x_test, {"preprocess": preprocess, "ncm_internal_l2": True}
    return apply_preprocess(x_train, x_test, preprocess)


def main():
    args = parse_args()

    def progress(message: str):
        if args.verbose_progress:
            print(f"[progress] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)

    root = project_root()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    backbone = canonical_backbone(args.backbone, args.feature_source)
    run_name = canonical_run_name(args.feature_source, args.run_name)
    result_dir = analysis_result_dir(
        output_dir, "layer_probe", args.dataset, backbone, run_name, args.normalization
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_json = result_dir / "metrics.json"
    result_csv = result_dir / "metrics.csv"
    if result_json.exists() and not args.overwrite:
        raise FileExistsError(f"Result file exists; use --overwrite to replace: {result_json}")
    if args.overwrite and result_csv.exists():
        result_csv.unlink()

    progress(f"load train split start dataset={args.dataset} backbone={backbone}")
    train = load_feature_split(output_dir, args.dataset, backbone, args.normalization, "train", run_name)
    progress(f"load train split done shards={len(train['shards'])} cls_shape={train['cls'].shape} mean_shape={train['mean'].shape}")
    progress(f"load test split start dataset={args.dataset} backbone={backbone}")
    test = load_feature_split(output_dir, args.dataset, backbone, args.normalization, "test", run_name)
    progress(f"load test split done shards={len(test['shards'])} cls_shape={test['cls'].shape} mean_shape={test['mean'].shape}")
    progress(f"labels train_class={train['class'].shape} test_class={test['class'].shape} train_domain={train['domain'].shape} test_domain={test['domain'].shape}")
    defaults = DATASET_DEFAULTS[args.dataset]
    domain_names = {}
    for meta in train.get("metadata", []) + test.get("metadata", []):
        if "domain_id" in meta and "domain_name" in meta:
            domain_names[int(meta["domain_id"])] = meta["domain_name"]

    rows = []
    details = {
        "dataset": args.dataset,
        "backbone": backbone,
        "normalization": args.normalization,
        "feature_source": args.feature_source,
        "run_name": run_name,
        "preprocess": args.preprocess,
        "train_shards": train["shards"],
        "test_shards": test["shards"],
        "results": [],
    }

    for feature_pool in args.feature_pools:
        x_train_all = train[feature_pool]
        x_test_all = test[feature_pool]
        if x_train_all.ndim != 3 or x_train_all.shape[1] != 12:
            raise RuntimeError(f"Expected train features [N,12,D], got {x_train_all.shape}")
        if x_test_all.ndim != 3 or x_test_all.shape[1] != 12:
            raise RuntimeError(f"Expected test features [N,12,D], got {x_test_all.shape}")
        for layer_idx in range(x_train_all.shape[1]):
            x_train_layer = x_train_all[:, layer_idx, :]
            x_test_layer = x_test_all[:, layer_idx, :]
            for target in args.targets:
                y_train = train[target]
                y_test = test[target]
                for classifier in args.classifiers:
                    step = f"dataset={args.dataset} backbone={backbone} pool={feature_pool} layer={layer_idx + 1:02d} target={target} clf={classifier}"
                    progress(f"step start {step} raw_train_shape={x_train_layer.shape} raw_test_shape={x_test_layer.shape}")
                    progress(f"preprocess start {step} mode={args.preprocess}")
                    x_train, x_test, preprocess_meta = preprocess_for_classifier(x_train_layer, x_test_layer, classifier, args.preprocess)
                    progress(f"preprocess done {step} train_shape={x_train.shape} test_shape={x_test.shape}")

                    def step_progress(message: str, _step=step):
                        progress(f"{message} {_step}")

                    if classifier == "linear":
                        y_pred, clf_meta = fit_predict_linear(
                            x_train,
                            y_train,
                            x_test,
                            dataset=args.dataset,
                            target=target,
                            method=args.linear_method,
                            seed=args.seed,
                            progress=step_progress,
                        )
                    elif classifier == "ncm":
                        progress(f"ncm predict start {step}")
                        y_pred, clf_meta = ncm_predict(x_train, y_train, x_test, normalize=True)
                        progress(f"ncm predict done {step}")
                    elif classifier == "wncm":
                        y_pred, clf_meta = whitened_ncm_predict(
                            x_train,
                            y_train,
                            x_test,
                            normalize=True,
                            reg=args.wncm_reg,
                            cov_max_samples=args.cov_max_samples,
                            seed=args.seed,
                            progress=step_progress,
                        )
                    else:
                        raise ValueError(classifier)

                    names = domain_names if target == "domain" else None
                    metrics = accuracy_metrics(y_test, y_pred, names)
                    group_metrics = class_group_accuracy(y_test, y_pred, defaults["init_cls"], defaults["increment"]) if target == "class" else {}
                    record = {
                        "dataset": args.dataset,
                        "backbone": backbone,
                        "normalization": args.normalization,
                        "feature_source": args.feature_source,
                        "run_name": run_name,
                        "feature_pool": feature_pool,
                        "layer": layer_idx + 1,
                        "target": target,
                        "classifier": classifier,
                        "preprocess": args.preprocess,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "num_train": int(len(y_train)),
                        "num_test": int(len(y_test)),
                        "classifier_meta": clf_meta,
                        "preprocess_meta": preprocess_meta,
                        "per_label_accuracy": metrics["per_label_accuracy"],
                        "class_group_accuracy": group_metrics,
                    }
                    details["results"].append(record)
                    row = {
                        "dataset": args.dataset,
                        "backbone": backbone,
                        "normalization": args.normalization,
                        "feature_source": args.feature_source,
                        "run_name": run_name,
                        "feature_pool": feature_pool,
                        "layer": layer_idx + 1,
                        "target": target,
                        "classifier": classifier,
                        "preprocess": args.preprocess,
                        "accuracy": f"{metrics['accuracy']:.6f}",
                        "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                        "num_train": int(len(y_train)),
                        "num_test": int(len(y_test)),
                        "linear_method": clf_meta.get("linear_method", ""),
                        "regularization": clf_meta.get("C", clf_meta.get("alpha", clf_meta.get("reg", ""))),
                    }
                    rows.append(row)
                    print(
                        f"[result] pool={feature_pool} layer={layer_idx + 1:02d} target={target} clf={classifier} acc={metrics['accuracy']:.2f} bal={metrics['balanced_accuracy']:.2f}",
                        flush=True,
                    )
                    if len(rows) >= 20:
                        append_csv(result_csv, rows, [
                            "dataset", "backbone", "normalization", "feature_source", "run_name", "feature_pool", "layer", "target", "classifier",
                            "preprocess", "accuracy", "balanced_accuracy", "num_train", "num_test", "linear_method", "regularization",
                        ])
                        rows = []
    if rows:
        append_csv(result_csv, rows, [
            "dataset", "backbone", "normalization", "feature_source", "run_name", "feature_pool", "layer", "target", "classifier",
            "preprocess", "accuracy", "balanced_accuracy", "num_train", "num_test", "linear_method", "regularization",
        ])
    atomic_write_json(result_json, details)
    print(f"[done] json={result_json}", flush=True)
    print(f"[done] csv={result_csv}", flush=True)


if __name__ == "__main__":
    main()
