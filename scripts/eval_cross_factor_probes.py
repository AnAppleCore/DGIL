import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
import numpy as np

from layer_probe_lib import (
    DATASET_DEFAULTS,
    accuracy_metrics,
    analysis_result_dir,
    apply_preprocess,
    atomic_write_json,
    canonical_backbone,
    canonical_run_name,
    fit_whitened_ncm,
    load_feature_layer,
    ncm_predict,
    predict_whitened_ncm,
    project_root,
)


PROTOCOLS = ("lodo_class", "heldout_task_domain")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate leakage-free cross-factor NCM/WNCM depth profiles.")
    parser.add_argument("--method", default="wncm", choices=["ncm", "wncm"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", default="results/layer_probe/raw_pretrained")
    parser.add_argument("--normalization", default="repo", choices=["repo", "imagenet", "clip"])
    parser.add_argument("--feature-source", default="raw_pretrained", choices=["raw_pretrained", "slca_trained"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--feature-pools", nargs="+", default=["cls", "mean"], choices=["cls", "mean"])
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--preprocess", default="standard", choices=["none", "standard"])
    parser.add_argument("--wncm-reg", type=float, default=1e-4)
    parser.add_argument("--cov-max-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--training-log-root", default="results/layer_probe/slca_trained/logs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-progress", action="store_true")
    return parser.parse_args()


def legacy_class_order(num_classes: int, seed: int) -> list[int]:
    state = np.random.RandomState(seed)
    return state.permutation(num_classes).tolist()


def task_class_sets(num_classes: int, init_cls: int, increment: int, seed: int) -> tuple[list[int], list[list[int]]]:
    order = legacy_class_order(num_classes, seed)
    sizes = [init_cls]
    while sum(sizes) + increment < num_classes:
        sizes.append(increment)
    remainder = num_classes - sum(sizes)
    if remainder > 0:
        sizes.append(remainder)
    tasks, offset = [], 0
    for size in sizes:
        tasks.append(order[offset:offset + size])
        offset += size
    return order, tasks


def find_logged_class_order(log_root: Path, dataset: str, num_classes: int) -> dict:
    if not log_root.exists():
        return {"found": False, "path": None, "class_order": None}
    dataset_pattern = re.compile(r"=> dataset: ([A-Za-z0-9_]+)")
    order_pattern = re.compile(r"Class order: (\[[^\n]+\])")
    candidates = []
    for path in sorted(log_root.glob("**/*.log")):
        current_dataset = None
        try:
            with path.open(errors="replace") as handle:
                for line in handle:
                    dataset_match = dataset_pattern.search(line)
                    if dataset_match:
                        current_dataset = dataset_match.group(1).lower()
                    order_match = order_pattern.search(line)
                    if order_match and current_dataset == dataset:
                        order = json.loads(order_match.group(1))
                        if len(order) == num_classes:
                            candidates.append((path, order))
        except OSError:
            continue
    if not candidates:
        return {"found": False, "path": None, "class_order": None}
    unique_orders = {tuple(order) for _, order in candidates}
    path, order = candidates[0]
    return {
        "found": True,
        "path": str(path),
        "class_order": order,
        "matching_log_count": len(candidates),
        "unique_logged_orders": len(unique_orders),
    }


def class_order_provenance(dataset: str, labels: np.ndarray, seed: int, log_root: Path) -> dict:
    num_classes = int(np.max(labels)) + 1
    defaults = DATASET_DEFAULTS[dataset]
    order, tasks = task_class_sets(num_classes, defaults["init_cls"], defaults["increment"], seed)
    logged = find_logged_class_order(log_root, dataset, num_classes)
    return {
        "class_order_source": "legacy_numpy_randomstate_permutation",
        "seed": int(seed),
        "num_classes": num_classes,
        "class_order": order,
        "task_class_sets": tasks,
        "training_log": logged,
        "validation_status": "matched" if logged.get("class_order") == order else ("not_found" if not logged.get("found") else "mismatch"),
    }


def result_directory(
    output_dir: Path,
    dataset: str,
    backbone: str,
    normalization: str,
    run_name: str,
    method: str,
    protocol: str,
) -> Path:
    return analysis_result_dir(
        output_dir,
        "cross_factor",
        dataset,
        backbone,
        run_name,
        normalization,
        method=method,
        protocol=protocol,
    )


def write_results(path: Path, payload: dict, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "metrics.json", payload)
    fieldnames = [
        "dataset", "backbone", "feature_source", "run_name", "normalization", "method", "protocol", "feature_pool", "layer",
        "holdout", "holdout_ids", "target", "accuracy", "balanced_accuracy", "num_train", "num_test",
        "target_supports", "cov_samples", "reg", "preprocess", "class_order_validation",
    ]
    csv_path = path / "metrics.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def preprocess_train_test(x_train: np.ndarray, x_test: np.ndarray, mode: str, method: str):
    if mode == "none":
        return x_train, x_test, {"preprocess": "none", "classifier_internal_l2": True, "method": method}
    train, test, meta = apply_preprocess(x_train, x_test, "standard")
    return train, test, {**meta, "classifier_internal_l2": True, "method": method}


def support_counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def per_class_domain_accuracy(class_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {}
    for class_id in np.unique(class_ids):
        mask = class_ids == class_id
        result[str(int(class_id))] = float((y_true[mask] == y_pred[mask]).mean() * 100.0)
    return result


def aggregate_records(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (record["feature_pool"], record["layer"])
        groups.setdefault(key, []).append(record)
    output = []
    for (pool, layer), group in sorted(groups.items()):
        balanced = np.array([r["balanced_accuracy"] for r in group], dtype=np.float64)
        accuracy = np.array([r["accuracy"] for r in group], dtype=np.float64)
        total_correct = sum(r["accuracy"] * r["num_test"] / 100.0 for r in group)
        total_samples = sum(r["num_test"] for r in group)
        worst = group[int(np.argmin(balanced))]
        output.append({
            "feature_pool": pool,
            "layer": int(layer),
            "holdout_macro_balanced_accuracy": float(balanced.mean()),
            "holdout_std_balanced_accuracy": float(balanced.std(ddof=0)),
            "holdout_macro_accuracy": float(accuracy.mean()),
            "micro_accuracy": float(total_correct / max(total_samples, 1) * 100.0),
            "worst_holdout": worst["holdout"],
            "worst_holdout_balanced_accuracy": float(worst["balanced_accuracy"]),
            "num_holdouts": len(group),
            "num_test": int(total_samples),
        })
    return output


def run_protocol(args, output_dir: Path, backbone: str, run_name: str, protocol: str, provenance: dict) -> tuple[dict, list[dict]]:
    records = []

    def progress(message: str):
        if args.verbose_progress:
            print(f"[progress] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)

    for pool in args.feature_pools:
        for layer in args.layers:
            progress(f"load pool={pool} layer={layer:02d}")
            train = load_feature_layer(output_dir, args.dataset, backbone, args.normalization, "train", pool, layer, run_name)
            test = load_feature_layer(output_dir, args.dataset, backbone, args.normalization, "test", pool, layer, run_name)
            train_classes, test_classes = train["class"], test["class"]
            train_domains, test_domains = train["domain"], test["domain"]
            if protocol == "lodo_class":
                holdouts = [(f"domain_{domain_id}", [int(domain_id)]) for domain_id in np.unique(test_domains)]
                target_name = "class"
            else:
                holdouts = [(f"task_{idx}", classes) for idx, classes in enumerate(provenance["task_class_sets"])]
                target_name = "domain"

            for holdout_name, holdout_ids in holdouts:
                if protocol == "lodo_class":
                    train_mask = ~np.isin(train_domains, holdout_ids)
                    test_mask = np.isin(test_domains, holdout_ids)
                    y_train, y_test = train_classes[train_mask], test_classes[test_mask]
                    covariance_factor = train_domains[train_mask]
                    fit_factor_values = np.unique(train_domains[train_mask])
                    test_factor_values = np.unique(test_domains[test_mask])
                else:
                    train_mask = ~np.isin(train_classes, holdout_ids)
                    test_mask = np.isin(test_classes, holdout_ids)
                    y_train, y_test = train_domains[train_mask], test_domains[test_mask]
                    covariance_factor = train_classes[train_mask]
                    fit_factor_values = np.unique(train_classes[train_mask])
                    test_factor_values = np.unique(test_classes[test_mask])
                if np.intersect1d(fit_factor_values, test_factor_values).size:
                    raise RuntimeError(f"Cross-factor leakage detected for {protocol}/{holdout_name}")
                if not len(y_train) or not len(y_test):
                    raise RuntimeError(f"Empty fit/test split for {protocol}/{holdout_name}")
                missing_targets = sorted(set(np.unique(y_test).tolist()) - set(np.unique(y_train).tolist()))
                if missing_targets:
                    raise RuntimeError(f"Test targets lack prototypes for {protocol}/{holdout_name}: {missing_targets}")

                x_train, x_test, preprocess_meta = preprocess_train_test(
                    train["features"][train_mask], test["features"][test_mask], args.preprocess, args.method
                )
                step = f"method={args.method} protocol={protocol} pool={pool} layer={layer:02d} holdout={holdout_name}"
                if args.method == "wncm":
                    model = fit_whitened_ncm(
                        x_train,
                        y_train,
                        normalize=True,
                        reg=args.wncm_reg,
                        cov_max_samples=args.cov_max_samples,
                        seed=args.seed,
                        covariance_cross_factors=covariance_factor,
                        progress=lambda message: progress(f"{message} {step}"),
                    )
                    y_pred = predict_whitened_ncm(model, x_test, progress=lambda message: progress(f"{message} {step}"))
                    classifier_metadata = model.metadata
                else:
                    y_pred, classifier_metadata = ncm_predict(x_train, y_train, x_test, normalize=True)
                metrics = accuracy_metrics(y_test, y_pred)
                detail = {
                    "dataset": args.dataset,
                    "backbone": backbone,
                    "feature_source": args.feature_source,
                    "run_name": run_name,
                    "normalization": args.normalization,
                    "method": args.method,
                    "protocol": protocol,
                    "feature_pool": pool,
                    "layer": int(layer),
                    "holdout": holdout_name,
                    "holdout_ids": [int(value) for value in holdout_ids],
                    "target": target_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "num_train": int(len(y_train)),
                    "num_test": int(len(y_test)),
                    "target_supports": support_counts(y_test),
                    "fit_target_supports": support_counts(y_train),
                    "per_target_accuracy": metrics["per_label_accuracy"],
                    "per_class_domain_accuracy": per_class_domain_accuracy(test_classes[test_mask], y_test, y_pred) if protocol == "heldout_task_domain" else {},
                    "fit_factor_values": [int(value) for value in fit_factor_values],
                    "test_factor_values": [int(value) for value in test_factor_values],
                    "preprocess_meta": preprocess_meta,
                    "classifier_metadata": classifier_metadata,
                }
                records.append(detail)
                if args.verbose_progress:
                    print(
                        f"[result] method={args.method} protocol={protocol} pool={pool} layer={layer:02d} holdout={holdout_name} "
                        f"acc={metrics['accuracy']:.2f} bal={metrics['balanced_accuracy']:.2f}",
                        flush=True,
                    )

    aggregates = aggregate_records(records)
    payload = {
        "dataset": args.dataset,
        "backbone": backbone,
        "feature_source": args.feature_source,
        "run_name": run_name,
        "normalization": args.normalization,
        "method": args.method,
        "protocol": protocol,
        "feature_pools": args.feature_pools,
        "layers": args.layers,
        "preprocess": args.preprocess,
        "classifier_parameters": {
            "method": args.method,
            "normalize": True,
            "reg": args.wncm_reg if args.method == "wncm" else None,
            "cov_max_samples": args.cov_max_samples if args.method == "wncm" else None,
            "seed": args.seed,
        },
        "class_order_provenance": provenance,
        "results": records,
        "aggregate": aggregates,
    }
    rows = []
    for record in records:
        rows.append({
            **record,
            "holdout_ids": json.dumps(record["holdout_ids"]),
            "target_supports": json.dumps(record["target_supports"], sort_keys=True),
            "cov_samples": record["classifier_metadata"].get("cov_samples"),
            "reg": args.wncm_reg if args.method == "wncm" else None,
            "preprocess": args.preprocess,
            "class_order_validation": provenance["validation_status"],
        })
    return payload, rows


def main():
    args = parse_args()
    invalid_layers = [layer for layer in args.layers if layer < 1 or layer > 12]
    if invalid_layers:
        raise ValueError(f"Layers must be in 1..12: {invalid_layers}")
    root = project_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    log_root = Path(args.training_log_root)
    if not log_root.is_absolute():
        log_root = (root / log_root).resolve()
    backbone = canonical_backbone(args.backbone, args.feature_source)
    run_name = canonical_run_name(args.feature_source, args.run_name)

    label_source = load_feature_layer(
        output_dir, args.dataset, backbone, args.normalization, "train", args.feature_pools[0], args.layers[0], run_name
    )
    provenance = class_order_provenance(args.dataset, label_source["class"], args.seed, log_root)
    if provenance["validation_status"] == "mismatch":
        raise RuntimeError(f"Reconstructed class order disagrees with training log: {provenance['training_log']['path']}")

    completed, skipped = [], []
    for protocol in args.protocols:
        destination = result_directory(output_dir, args.dataset, backbone, args.normalization, run_name, args.method, protocol)
        result_json = destination / "metrics.json"
        if result_json.exists() and not args.overwrite:
            print(f"[skip] protocol result exists: {result_json}", flush=True)
            skipped.append(protocol)
            continue
        payload, rows = run_protocol(args, output_dir, backbone, run_name, protocol, provenance)
        write_results(destination, payload, rows)
        completed.append(protocol)
        print(f"[done] protocol={protocol} json={result_json}", flush=True)
    if not completed and skipped:
        print(f"[done] all requested protocols already exist: {', '.join(skipped)}", flush=True)


if __name__ == "__main__":
    main()
