import argparse
import json
import subprocess
import time
from pathlib import Path

import torch


DATASETS = ["digitsdg", "officehome", "core50", "domainnet"]
ALIASES = ["default", "ibot", "mae", "dinov2", "clip"]
SLCA_BACKBONES = {
    "default": "vit_base_patch16_224",
    "ibot": "vit_base_patch16_224_21k_ibot",
    "mae": "vit_base_patch16_224_mae",
    "dinov2": "vit_base_patch14_224_dinov2",
    "clip": "vit_base_patch16_224_clip",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch trained SLCA feature extraction/probe jobs after checkpoints exist.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--backbones", nargs="+", default=ALIASES, choices=ALIASES)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--gpus", default="7", help="Comma-separated GPU ids. One serial worker is launched per GPU.")
    parser.add_argument("--output-dir", default="results/layer_probe/slca_trained")
    parser.add_argument("--normalization", default="repo", choices=["repo", "imagenet", "clip"])
    parser.add_argument("--run-name", default="seed1994_final")
    parser.add_argument("--session-prefix", default="slca_probe")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--linear-method", default="ridge", choices=["auto", "logreg", "ridge", "sgd"])
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wait", action="store_true", help="Create a watcher that waits for missing checkpoints before launching jobs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_path(output_dir: Path, dataset: str, alias: str, seed: int) -> Path:
    backbone = SLCA_BACKBONES[alias]
    return output_dir / "checkpoints" / dataset / safe_name(backbone) / f"seed{seed}" / "final.pkl"


def result_path(root: Path, output_dir_arg: str, dataset: str, alias: str, run_name: str, normalization: str) -> Path:
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir / "analyses" / "layer_probe" / dataset / safe_name(SLCA_BACKBONES[alias]) / safe_name(run_name) / normalization / "metrics.json"


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return f".{name}\t" in result.stdout or f".{name} " in result.stdout


def validate_checkpoint(path: Path, dataset: str, alias: str, seed: int) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint missing non-empty model_state_dict: {path}")
    expected_backbone = SLCA_BACKBONES[alias]
    errors = []
    if payload.get("dataset") != dataset:
        errors.append(f"dataset={payload.get('dataset')} expected={dataset}")
    if payload.get("backbone_type") != expected_backbone:
        errors.append(f"backbone={payload.get('backbone_type')} expected={expected_backbone}")
    if int(payload.get("seed", -1)) != int(seed):
        errors.append(f"seed={payload.get('seed')} expected={seed}")
    key_text = "\n".join(state.keys())
    required = ["backbone.patch_embed", "backbone.blocks.0", "backbone.blocks.11"]
    missing_prefixes = [prefix for prefix in required if prefix not in key_text]
    if missing_prefixes:
        errors.append(f"missing_backbone_prefixes={missing_prefixes}")
    if errors:
        raise RuntimeError(f"Invalid checkpoint {path}: " + "; ".join(errors))
    return {
        "path": str(path),
        "dataset": dataset,
        "alias": alias,
        "backbone": expected_backbone,
        "seed": seed,
        "tasks": payload.get("tasks"),
        "known_classes": payload.get("known_classes"),
        "total_classes": payload.get("total_classes"),
        "state_keys": len(state),
    }


def build_probe_commands(args, dataset: str, alias: str, ckpt: Path) -> tuple[str, str]:
    extract = (
        f"conda run -n DGIL python scripts/extract_layerwise_features.py "
        f"--dataset {dataset} --backbone {alias} --output-dir {args.output_dir} --normalization {args.normalization} "
        f"--feature-source slca_trained --checkpoint-path '{ckpt}' --run-name {args.run_name} "
        f"--device 0 --batch-size {args.batch_size} --num-workers {args.num_workers}"
    )
    classifiers = "ncm wncm" if dataset == "domainnet" else "linear ncm wncm"
    linear_method = args.linear_method
    eval_cmd = (
        f"conda run -n DGIL python scripts/eval_layerwise_feature_probes.py "
        f"--dataset {dataset} --backbone {alias} --output-dir {args.output_dir} --normalization {args.normalization} "
        f"--feature-source slca_trained --run-name {args.run_name} --classifiers {classifiers} "
        f"--linear-method {linear_method} --verbose-progress --overwrite"
    )
    return extract, eval_cmd


def main():
    args = parse_args()
    root = project_root()
    out = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    log_root = out / "logs" / time.strftime("trained_probe_%Y%m%d_%H%M%S")
    log_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    tasks = [(d, a, checkpoint_path(out, d, a, args.seed)) for d in args.datasets for a in args.backbones]
    existing_validations = []
    for dataset, alias, ckpt in tasks:
        if ckpt.exists():
            existing_validations.append(validate_checkpoint(ckpt, dataset, alias, args.seed))
    validation_path = log_root / "checkpoint_validation_existing.json"
    validation_path.write_text(json.dumps(existing_validations, indent=2, sort_keys=True) + "\n")
    print(f"[checkpoint-validation] existing={len(existing_validations)}/{len(tasks)} path={validation_path}", flush=True)

    missing = [str(p) for _, _, p in tasks if not p.exists()]
    if missing and not args.wait:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    if args.wait and missing:
        watcher = log_root / "wait_and_launch.sh"
        rerun = (
            f"cd '{root}' && conda run -n DGIL python scripts/run_slca_trained_layer_probes.py "
            f"--datasets {' '.join(args.datasets)} --backbones {' '.join(args.backbones)} --seed {args.seed} "
            f"--gpus {','.join(gpus)} --output-dir {args.output_dir} --normalization {args.normalization} "
            f"--run-name {args.run_name} --session-prefix {args.session_prefix} --batch-size {args.batch_size} "
            f"--num-workers {args.num_workers} --linear-method {args.linear_method} --cpu-threads {args.cpu_threads}"
        )
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"echo '[watcher-start] {time.ctime()}'", "while true; do", "  missing=0"]
        for _, _, p in tasks:
            lines.append(f"  [[ -f '{p}' ]] || missing=$((missing+1))")
        lines += [
            "  echo \"[watcher] missing=${missing} time=$(date)\"",
            "  [[ ${missing} -eq 0 ]] && break",
            "  sleep 300",
            "done",
            "echo '[watcher] all checkpoints found; validating and launching probes' $(date)",
            rerun,
            "echo '[watcher-done]' $(date)",
        ]
        watcher.write_text("\n".join(lines) + "\n")
        watcher.chmod(0o755)
        session = f"{args.session_prefix}_watch"
        log = log_root / "watcher.log"
        if not args.dry_run:
            subprocess.run(["screen", "-dmS", session, "bash", "-lc", f"'{watcher}' > '{log}' 2>&1"], check=True)
        print(f"[watcher] session={session} log={log}", flush=True)
        return

    all_validations = [validate_checkpoint(ckpt, dataset, alias, args.seed) for dataset, alias, ckpt in tasks]
    all_validation_path = log_root / "checkpoint_validation_all.json"
    all_validation_path.write_text(json.dumps(all_validations, indent=2, sort_keys=True) + "\n")
    print(f"[checkpoint-validation] all={len(all_validations)}/{len(tasks)} path={all_validation_path}", flush=True)

    pending_tasks = []
    for dataset, alias, ckpt in tasks:
        result_json = result_path(root, args.output_dir, dataset, alias, args.run_name, args.normalization)
        if result_json.exists():
            print(f"[skip] result exists dataset={dataset} alias={alias} path={result_json}", flush=True)
            continue
        pending_tasks.append((dataset, alias, ckpt))

    queues = {gpu: [] for gpu in gpus}
    for idx, task in enumerate(pending_tasks):
        queues[gpus[idx % len(gpus)]].append(task)

    launched = []
    for gpu, gpu_tasks in queues.items():
        if not gpu_tasks:
            continue
        session = f"{args.session_prefix}_gpu{gpu}"
        if screen_exists(session):
            print(f"[skip] screen exists {session}", flush=True)
            continue
        worker = log_root / f"worker_gpu{gpu}.sh"
        log = log_root / f"worker_gpu{gpu}.log"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd '{root}'",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
            f"export OMP_NUM_THREADS={args.cpu_threads}",
            f"export MKL_NUM_THREADS={args.cpu_threads}",
            f"export OPENBLAS_NUM_THREADS={args.cpu_threads}",
            f"export NUMEXPR_NUM_THREADS={args.cpu_threads}",
            f"echo '[worker-start] gpu={gpu} time='$(date)",
        ]
        manifest_tasks = []
        for dataset, alias, ckpt in gpu_tasks:
            result_json = result_path(root, args.output_dir, dataset, alias, args.run_name, args.normalization)
            extract, eval_cmd = build_probe_commands(args, dataset, alias, ckpt)
            lines.extend([
                f"if [[ -f '{result_json}' ]]; then echo '[skip-existing] dataset={dataset} alias={alias} result={result_json}'; else",
                f"  echo '[probe-start] dataset={dataset} alias={alias} gpu={gpu} time='$(date)",
                f"  {extract}",
                f"  {eval_cmd}",
                f"  echo '[probe-done] dataset={dataset} alias={alias} gpu={gpu} time='$(date)",
                "fi",
            ])
            manifest_tasks.append({"dataset": dataset, "alias": alias, "checkpoint": str(ckpt), "result": str(result_json)})
        lines.append(f"echo '[worker-done] gpu={gpu} time='$(date)")
        worker.write_text("\n".join(lines) + "\n")
        worker.chmod(0o755)
        print(f"[launch] {session} gpu={gpu} tasks={len(gpu_tasks)} log={log}", flush=True)
        if not args.dry_run:
            subprocess.run(["screen", "-dmS", session, "bash", "-lc", f"'{worker}' > '{log}' 2>&1"], check=True)
        launched.append({"session": session, "gpu": gpu, "worker": str(worker), "log": str(log), "tasks": manifest_tasks})

    manifest = log_root / "manifest.json"
    manifest.write_text(json.dumps({"launched": launched, "num_pending_tasks": len(pending_tasks)}, indent=2, sort_keys=True) + "\n")
    print(f"[done] manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
