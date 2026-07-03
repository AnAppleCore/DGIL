import argparse
import json
import subprocess
import time
from pathlib import Path


DATASETS = ["digitsdg", "officehome", "core50", "domainnet"]
CONFIGS = {
    "default": "slca_dgil.json",
    "ibot": "pretrain_ibot_21k/slca_dgil_ibot_21k.json",
    "mae": "pretrain_mae/slca_dgil_mae.json",
    "dinov2": "pretrain_dinov2/slca_dgil_dinov2.json",
    "clip": "pretrain_clip/slca_dgil_clip.json",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch one-seed SLCA DGIL final-checkpoint training jobs in screen queues.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--backbones", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--gpus", default="7", help="Comma-separated GPU ids. One serial worker is launched per GPU.")
    parser.add_argument("--output-dir", default="results/layer_probe_slca_trained")
    parser.add_argument("--session-prefix", default="slca_ckpt")
    parser.add_argument("--pilot", action="store_true", help="Only run digitsdg/clip unless explicit datasets/backbones are provided.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return f".{name}\t" in result.stdout or f".{name} " in result.stdout


def main():
    args = parse_args()
    root = project_root()
    out = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    config_root = out / "configs"
    log_root = out / "logs" / time.strftime("train_%Y%m%d_%H%M%S")
    checkpoint_root = out / "checkpoints"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    datasets = list(args.datasets)
    backbones = list(args.backbones)
    if args.pilot:
        datasets = ["digitsdg"]
        backbones = ["clip"]

    tasks = []
    for dataset in datasets:
        for alias in backbones:
            src = root / "configs" / "DGIL" / dataset / CONFIGS[alias]
            if not src.exists():
                raise FileNotFoundError(src)
            cfg = json.loads(src.read_text())
            cfg["seed"] = [args.seed]
            cfg["device"] = ["0"]
            cfg["prefix"] = "DGIL_slca_probe_ckpt"
            cfg["save_final_checkpoint"] = True
            cfg["checkpoint_dir"] = str(checkpoint_root)
            dst_dir = config_root / dataset
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{alias}_seed{args.seed}.json"
            dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
            ckpt = checkpoint_root / dataset / safe_name(cfg["backbone_type"]) / f"seed{args.seed}" / "final.pkl"
            if ckpt.exists():
                print(f"[skip] checkpoint exists dataset={dataset} alias={alias} path={ckpt}", flush=True)
                continue
            tasks.append({
                "dataset": dataset,
                "alias": alias,
                "config": str(dst),
                "checkpoint": str(ckpt),
            })

    queues = {gpu: [] for gpu in gpus}
    for idx, task in enumerate(tasks):
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
            f"echo '[worker-start] gpu={gpu} time='$(date)",
        ]
        for task in gpu_tasks:
            dataset = task["dataset"]
            alias = task["alias"]
            cfg_path = task["config"]
            ckpt_path = task["checkpoint"]
            lines.extend([
                f"if [[ -f '{ckpt_path}' ]]; then echo '[skip-existing] dataset={dataset} alias={alias} checkpoint={ckpt_path}'; else",
                f"  echo '[train-start] dataset={dataset} alias={alias} gpu={gpu} time='$(date)",
                f"  conda run -n DGIL python main.py --config '{cfg_path}'",
                f"  echo '[train-done] dataset={dataset} alias={alias} gpu={gpu} time='$(date)",
                "fi",
            ])
        lines.append(f"echo '[worker-done] gpu={gpu} time='$(date)")
        worker.write_text("\n".join(lines) + "\n")
        worker.chmod(0o755)
        print(f"[launch] {session} gpu={gpu} tasks={len(gpu_tasks)} log={log}", flush=True)
        if not args.dry_run:
            subprocess.run(["screen", "-dmS", session, "bash", "-lc", f"'{worker}' > '{log}' 2>&1"], check=True)
        launched.append({"session": session, "gpu": gpu, "worker": str(worker), "log": str(log), "tasks": gpu_tasks})

    manifest = log_root / "manifest.json"
    manifest.write_text(json.dumps({"launched": launched, "checkpoint_root": str(checkpoint_root), "num_pending_tasks": len(tasks)}, indent=2, sort_keys=True) + "\n")
    print(f"[done] manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
