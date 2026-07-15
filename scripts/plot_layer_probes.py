import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_ORDER = ["digitsdg", "officehome", "core50", "domainnet"]
TARGET_ORDER = ["class", "domain"]
CLASSIFIER_ORDER = ["linear", "ncm", "wncm"]
POOL_STYLES = {
    "cls": {"color": "#1f77b4", "marker": "o"},
    "mean": {"color": "#d62728", "marker": "s"},
}
BACKBONE_ALIASES = {
    "vit_base_patch16_224_dot": "default",
    "vit_base_patch16_224": "default",
    "vit_base_patch16_224_21k_ibot_dot": "iBOT-21K",
    "vit_base_patch16_224_21k_ibot": "iBOT-21K",
    "vit_base_patch16_224_mae_dot": "MAE",
    "vit_base_patch16_224_mae": "MAE",
    "vit_base_patch14_224_dinov2_dot": "DINOv2",
    "vit_base_patch14_224_dinov2": "DINOv2",
    "vit_base_patch16_224_clip_dot": "CLIP",
    "vit_base_patch16_224_clip": "CLIP",
}
BACKBONE_ORDER = ["default", "iBOT-21K", "MAE", "DINOv2", "CLIP"]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot layer-wise feature probes.")
    parser.add_argument("--results-root", default="results/layer_probe")
    parser.add_argument("--feature-source", required=True, choices=["raw_pretrained", "slca_trained"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--normalization", default="repo")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def curve_summary(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["target"], row["classifier"], row["feature_pool"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item["layer"])
        first, final = group[0], group[-1]
        peak = max(group, key=lambda item: item["balanced_accuracy"])
        summaries.append({
            "dataset": key[0],
            "backbone": key[1],
            "target": key[2],
            "classifier": key[3],
            "feature_pool": key[4],
            "first_layer": first["layer"],
            "first_balanced_accuracy": first["balanced_accuracy"],
            "peak_layer": peak["layer"],
            "peak_balanced_accuracy": peak["balanced_accuracy"],
            "final_layer": final["layer"],
            "final_balanced_accuracy": final["balanced_accuracy"],
            "final_minus_first": final["balanced_accuracy"] - first["balanced_accuracy"],
            "peak_minus_final": peak["balanced_accuracy"] - final["balanced_accuracy"],
        })
    return summaries


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    source_dir = results_root / args.feature_source
    run_name = args.run_name or ("seed1994_final" if args.feature_source == "slca_trained" else "default")
    analysis_dir = source_dir / "analyses" / "layer_probe"
    figure_dir = source_dir / "figures" / "layer_probe"
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(analysis_dir.glob(f"*/*/{run_name}/{args.normalization}/metrics.json"))
    if not paths:
        raise SystemExit(f"No layer-probe results found under {analysis_dir}")

    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        backbone_name = payload["backbone"]
        backbone_label = BACKBONE_ALIASES.get(backbone_name, backbone_name)
        for record in payload.get("results", []):
            rows.append({
                "dataset": payload["dataset"],
                "backbone": backbone_label,
                "backbone_name": backbone_name,
                "feature_source": payload.get("feature_source", args.feature_source),
                "run_name": payload.get("run_name", run_name),
                "normalization": payload["normalization"],
                "feature_pool": record["feature_pool"],
                "layer": int(record["layer"]),
                "target": record["target"],
                "classifier": record["classifier"],
                "accuracy": float(record["accuracy"]),
                "balanced_accuracy": float(record["balanced_accuracy"]),
                "num_train": int(record["num_train"]),
                "num_test": int(record["num_test"]),
            })

    profile_fields = [
        "dataset", "backbone", "backbone_name", "feature_source", "run_name", "normalization",
        "feature_pool", "layer", "target", "classifier", "accuracy", "balanced_accuracy", "num_train", "num_test",
    ]
    summary_rows = curve_summary(rows)
    write_csv(figure_dir / "depth_profiles.csv", rows, profile_fields)
    write_csv(figure_dir / "curve_summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else [])

    datasets = [dataset for dataset in DATASET_ORDER if any(row["dataset"] == dataset for row in rows)]
    backbones = [backbone for backbone in BACKBONE_ORDER if any(row["backbone"] == backbone for row in rows)]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 140, "savefig.dpi": 180})

    for target in TARGET_ORDER:
        for classifier in CLASSIFIER_ORDER:
            selected = [row for row in rows if row["target"] == target and row["classifier"] == classifier]
            if not selected:
                continue
            fig, axes = plt.subplots(
                len(datasets),
                len(backbones),
                figsize=(3.5 * len(backbones), 2.8 * len(datasets)),
                squeeze=False,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for row_index, dataset in enumerate(datasets):
                for column_index, backbone in enumerate(backbones):
                    ax = axes[row_index][column_index]
                    subset = [
                        row for row in selected
                        if row["dataset"] == dataset and row["backbone"] == backbone
                    ]
                    if not subset:
                        ax.text(0.5, 0.5, "not evaluated", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                        continue
                    for pool, style in POOL_STYLES.items():
                        curve = sorted(
                            (row for row in subset if row["feature_pool"] == pool),
                            key=lambda item: item["layer"],
                        )
                        if not curve:
                            continue
                        ax.plot(
                            [row["layer"] for row in curve],
                            [row["balanced_accuracy"] for row in curve],
                            color=style["color"],
                            marker=style["marker"],
                            markersize=3,
                            linewidth=1.5,
                            label=pool,
                        )
                    ax.set_title(f"{dataset} / {backbone}")
                    ax.set_xticks(range(1, 13))
                    ax.set_ylim(0, 100)
                    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
                    if row_index == len(datasets) - 1:
                        ax.set_xlabel("Layer")
                    if column_index == 0:
                        ax.set_ylabel("Balanced accuracy (%)")
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
                f"{args.feature_source} / {classifier.upper()}: {target} prediction",
                y=1.065,
                fontsize=14,
                fontweight="bold",
            )
            fig.savefig(figure_dir / f"{target}_{classifier}.png", bbox_inches="tight", pad_inches=0.18)
            plt.close(fig)

    print(f"[done] result_files={len(paths)} rows={len(rows)} summaries={len(summary_rows)} figures={figure_dir}")


if __name__ == "__main__":
    main()
