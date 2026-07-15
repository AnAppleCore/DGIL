import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTOCOL_LABELS = {
    "lodo_class": "Class prediction across held-out domains",
    "heldout_task_domain": "Domain prediction across held-out semantic tasks",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot cross-factor NCM/WNCM depth profiles.")
    parser.add_argument("--method", default="wncm", choices=["ncm", "wncm"])
    parser.add_argument("--results-root", default="results/layer_probe")
    parser.add_argument("--feature-source", required=True, choices=["slca_trained", "raw_pretrained"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--normalization", default="repo")
    return parser.parse_args()


def discover_results(source_dir: Path, run_name: str, normalization: str, method: str) -> list[Path]:
    pattern = f"*/*/{run_name}/{normalization}/*/metrics.json"
    return sorted((source_dir / "analyses" / "cross_factor" / method).glob(pattern))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def curve_summary(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["feature_source"], row["protocol"], row["feature_pool"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item["layer"])
        first, final = group[0], group[-1]
        peak = max(group, key=lambda item: item["holdout_macro_balanced_accuracy"])
        worst = min(group, key=lambda item: item["worst_holdout_balanced_accuracy"])
        summaries.append({
            "dataset": key[0],
            "backbone": key[1],
            "feature_source": key[2],
            "protocol": key[3],
            "feature_pool": key[4],
            "first_layer": first["layer"],
            "first_balanced_accuracy": first["holdout_macro_balanced_accuracy"],
            "peak_layer": peak["layer"],
            "peak_balanced_accuracy": peak["holdout_macro_balanced_accuracy"],
            "final_layer": final["layer"],
            "final_balanced_accuracy": final["holdout_macro_balanced_accuracy"],
            "final_minus_first": final["holdout_macro_balanced_accuracy"] - first["holdout_macro_balanced_accuracy"],
            "peak_minus_final": peak["holdout_macro_balanced_accuracy"] - final["holdout_macro_balanced_accuracy"],
            "worst_layer": worst["layer"],
            "worst_holdout": worst["worst_holdout"],
            "worst_holdout_balanced_accuracy": worst["worst_holdout_balanced_accuracy"],
            "max_holdout_std": max(item["holdout_std_balanced_accuracy"] for item in group),
        })
    return summaries


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    source_dir = results_root / args.feature_source
    run_name = args.run_name or ("seed1994_final" if args.feature_source == "slca_trained" else "default")
    figure_dir = source_dir / "figures" / "cross_factor" / args.method
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_results(source_dir, run_name, args.normalization, args.method)
    if not paths:
        raise SystemExit(f"No cross-factor results found under {source_dir}")

    aggregate_rows, holdout_rows = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        common = {
            "dataset": payload["dataset"],
            "backbone": payload["backbone"],
            "feature_source": payload["feature_source"],
            "run_name": payload.get("run_name"),
            "normalization": payload["normalization"],
            "method": payload.get("method", args.method),
            "protocol": payload["protocol"],
        }
        for row in payload.get("aggregate", []):
            aggregate_rows.append({**common, **row})
        for row in payload.get("results", []):
            holdout_rows.append({
                **common,
                "feature_pool": row["feature_pool"],
                "layer": row["layer"],
                "holdout": row["holdout"],
                "holdout_ids": json.dumps(row["holdout_ids"]),
                "accuracy": row["accuracy"],
                "balanced_accuracy": row["balanced_accuracy"],
                "num_train": row["num_train"],
                "num_test": row["num_test"],
            })

    aggregate_fields = [
        "dataset", "backbone", "feature_source", "run_name", "normalization", "method", "protocol", "feature_pool", "layer",
        "holdout_macro_balanced_accuracy", "holdout_std_balanced_accuracy", "holdout_macro_accuracy", "micro_accuracy",
        "worst_holdout", "worst_holdout_balanced_accuracy", "num_holdouts", "num_test",
    ]
    holdout_fields = [
        "dataset", "backbone", "feature_source", "run_name", "normalization", "method", "protocol", "feature_pool", "layer",
        "holdout", "holdout_ids", "accuracy", "balanced_accuracy", "num_train", "num_test",
    ]
    summary_rows = curve_summary(aggregate_rows)
    summary_fields = list(summary_rows[0]) if summary_rows else []
    write_csv(figure_dir / "depth_profiles.csv", aggregate_rows, aggregate_fields)
    write_csv(figure_dir / "holdout_metrics.csv", holdout_rows, holdout_fields)
    if summary_rows:
        write_csv(figure_dir / "curve_summary.csv", summary_rows, summary_fields)

    datasets = sorted({row["dataset"] for row in aggregate_rows})
    backbones = sorted({row["backbone"] for row in aggregate_rows})
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 140, "savefig.dpi": 180})
    for protocol in sorted({row["protocol"] for row in aggregate_rows}):
        fig, axes = plt.subplots(
            len(datasets), len(backbones), figsize=(3.5 * len(backbones), 2.8 * len(datasets)), squeeze=False,
            sharex=True, sharey=True, constrained_layout=True,
        )
        for row_index, dataset in enumerate(datasets):
            for column_index, backbone in enumerate(backbones):
                ax = axes[row_index][column_index]
                subset = [row for row in aggregate_rows if row["protocol"] == protocol and row["dataset"] == dataset and row["backbone"] == backbone]
                if not subset:
                    ax.set_axis_off()
                    continue
                for pool, color, marker in (("cls", "#1f77b4", "o"), ("mean", "#d62728", "s")):
                    curve = sorted((row for row in subset if row["feature_pool"] == pool), key=lambda item: item["layer"])
                    if not curve:
                        continue
                    x = np.array([row["layer"] for row in curve])
                    y = np.array([row["holdout_macro_balanced_accuracy"] for row in curve])
                    std = np.array([row["holdout_std_balanced_accuracy"] for row in curve])
                    ax.plot(x, y, color=color, marker=marker, markersize=3, linewidth=1.5, label=pool)
                    ax.fill_between(x, np.maximum(0, y - std), np.minimum(100, y + std), color=color, alpha=0.12)
                ax.set_title(f"{dataset} / {backbone}")
                ax.set_xticks(range(1, 13))
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
                if row_index == len(datasets) - 1:
                    ax.set_xlabel("Layer")
                if column_index == 0:
                    ax.set_ylabel("Holdout-macro balanced accuracy (%)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.035),
                ncol=2,
                frameon=False,
                fontsize=12,
            )
        fig.suptitle(
            f"{args.feature_source} / {args.method.upper()}: {PROTOCOL_LABELS[protocol]}",
            y=1.065,
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(figure_dir / f"{protocol}.png", bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)

    print(f"[done] result_files={len(paths)} aggregate_rows={len(aggregate_rows)} holdout_rows={len(holdout_rows)} figures={figure_dir}")


if __name__ == "__main__":
    main()
