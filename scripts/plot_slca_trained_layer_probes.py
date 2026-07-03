#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/data2/hongwei/DGIL/results/layer_probe_slca_trained")
OUT = ROOT / "figures" / "seed1994_final"
OUT.mkdir(parents=True, exist_ok=True)

BACKBONE_LABELS = {
    "vit_base_patch16_224": "default",
    "vit_base_patch16_224_21k_ibot": "iBOT-21K",
    "vit_base_patch16_224_mae": "MAE",
    "vit_base_patch14_224_dinov2": "DINOv2",
    "vit_base_patch16_224_clip": "CLIP",
}
BACKBONE_ORDER = ["default", "iBOT-21K", "MAE", "DINOv2", "CLIP"]
POOL_ORDER = ["cls", "mean"]
TARGET_ORDER = ["class", "domain"]
DATASET_ORDER = ["digitsdg", "officehome", "core50", "domainnet"]
CLASSIFIER_ORDER = ["linear", "ncm", "wncm"]
COLORS = {"linear": "#1f77b4", "ncm": "#ff7f0e", "wncm": "#2ca02c"}
MARKERS = {"linear": "o", "ncm": "s", "wncm": "^"}

rows = []
for path in sorted((ROOT / "probe_results").glob("*/*/seed1994_final/repo/results.json")):
    payload = json.loads(path.read_text())
    backbone = BACKBONE_LABELS.get(payload.get("backbone"), payload.get("backbone", "unknown"))
    dataset = payload.get("dataset", path.parts[-6])
    for r in payload.get("results", []):
        rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "pool": r.get("feature_pool"),
            "layer": int(r.get("layer")),
            "target": r.get("target"),
            "classifier": r.get("classifier"),
            "accuracy": float(r.get("accuracy")),
            "balanced_accuracy": float(r.get("balanced_accuracy")),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No results found")

# Save tidy data for later reuse.
df.to_csv(OUT / "slca_trained_layer_probe_tidy.csv", index=False)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 180,
})

summary_rows = []

for dataset in DATASET_ORDER:
    for target in TARGET_ORDER:
        sub_dt = df[(df["dataset"] == dataset) & (df["target"] == target)].copy()
        if sub_dt.empty:
            continue
        fig, axes = plt.subplots(
            nrows=len(BACKBONE_ORDER),
            ncols=len(POOL_ORDER),
            figsize=(13.5, 16.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for i, backbone in enumerate(BACKBONE_ORDER):
            for j, pool in enumerate(POOL_ORDER):
                ax = axes[i, j]
                sub = sub_dt[(sub_dt["backbone"] == backbone) & (sub_dt["pool"] == pool)]
                if sub.empty:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                for clf in CLASSIFIER_ORDER:
                    s = sub[sub["classifier"] == clf].sort_values("layer")
                    if s.empty:
                        continue
                    ax.plot(
                        s["layer"],
                        s["accuracy"],
                        label=clf,
                        color=COLORS[clf],
                        marker=MARKERS[clf],
                        linewidth=1.8,
                        markersize=4,
                        alpha=0.95,
                    )
                    peak_idx = s["accuracy"].idxmax()
                    peak = s.loc[peak_idx]
                    final = s[s["layer"] == s["layer"].max()].iloc[0]
                    first = s[s["layer"] == s["layer"].min()].iloc[0]
                    summary_rows.append({
                        "dataset": dataset,
                        "target": target,
                        "backbone": backbone,
                        "pool": pool,
                        "classifier": clf,
                        "first_layer": int(first["layer"]),
                        "first_acc": first["accuracy"],
                        "peak_layer": int(peak["layer"]),
                        "peak_acc": peak["accuracy"],
                        "final_layer": int(final["layer"]),
                        "final_acc": final["accuracy"],
                        "peak_minus_final": peak["accuracy"] - final["accuracy"],
                        "final_minus_first": final["accuracy"] - first["accuracy"],
                    })
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
                ax.set_title(f"{backbone} / {pool}")
                if i == len(BACKBONE_ORDER) - 1:
                    ax.set_xlabel("Layer")
                if j == 0:
                    ax.set_ylabel("Accuracy (%)")
                ax.set_xticks(sorted(sub["layer"].unique()))
                ax.set_ylim(0, min(100, max(5, sub_dt["accuracy"].max() * 1.08)))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f"SLCA-trained layer probes: {dataset} / target={target}", y=1.025, fontsize=15, fontweight="bold")
        out = OUT / f"slca_trained_{dataset}_{target}_by_backbone_pool_classifier.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

summary = pd.DataFrame(summary_rows)
if not summary.empty:
    summary.to_csv(OUT / "slca_trained_peak_summary.csv", index=False)

# Overview figure: one page per target with all datasets, best classifier per dataset/backbone/pool retained as separate lines.
for target in TARGET_ORDER:
    sub_t = df[df["target"] == target]
    if sub_t.empty:
        continue
    fig, axes = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=len(BACKBONE_ORDER),
        figsize=(19, 13),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for i, dataset in enumerate(DATASET_ORDER):
        for j, backbone in enumerate(BACKBONE_ORDER):
            ax = axes[i, j]
            sub = sub_t[(sub_t["dataset"] == dataset) & (sub_t["backbone"] == backbone)]
            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            for pool in POOL_ORDER:
                for clf in CLASSIFIER_ORDER:
                    s = sub[(sub["pool"] == pool) & (sub["classifier"] == clf)].sort_values("layer")
                    if s.empty:
                        continue
                    ax.plot(
                        s["layer"], s["accuracy"],
                        label=f"{pool}-{clf}",
                        linewidth=1.2,
                        alpha=0.9,
                    )
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
            ax.set_title(f"{dataset} / {backbone}")
            if i == len(DATASET_ORDER) - 1:
                ax.set_xlabel("Layer")
            if j == 0:
                ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(f"SLCA-trained overview: target={target}", y=1.035, fontsize=16, fontweight="bold")
    fig.savefig(OUT / f"slca_trained_overview_target_{target}.png", bbox_inches="tight")
    plt.close(fig)

print(f"rows={len(df)}")
print(f"figures_dir={OUT}")
for p in sorted(OUT.glob("*.png")):
    print(p)
