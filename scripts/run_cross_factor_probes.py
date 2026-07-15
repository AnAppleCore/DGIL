import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


DATASETS = ["digitsdg", "officehome", "core50", "domainnet"]
BACKBONES = ["default", "ibot", "mae", "dinov2", "clip"]
BACKBONE_NAMES = {
    "raw_pretrained": {
        "default": "vit_base_patch16_224_dot",
        "ibot": "vit_base_patch16_224_21k_ibot_dot",
        "mae": "vit_base_patch16_224_mae_dot",
        "dinov2": "vit_base_patch14_224_dinov2_dot",
        "clip": "vit_base_patch16_224_clip_dot",
    },
    "slca_trained": {
        "default": "vit_base_patch16_224",
        "ibot": "vit_base_patch16_224_21k_ibot",
        "mae": "vit_base_patch16_224_mae",
        "dinov2": "vit_base_patch14_224_dinov2",
        "clip": "vit_base_patch16_224_clip",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch resumable cross-factor NCM/WNCM jobs.")
    parser.add_argument("--method", default="wncm", choices=["ncm", "wncm"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    parser.add_argument("--feature-sources", nargs="+", default=["slca_trained"], choices=["slca_trained", "raw_pretrained"])
    parser.add_argument("--protocols", nargs="+", default=["lodo_class", "heldout_task_domain"], choices=["lodo_class", "heldout_task_domain"])
    parser.add_argument("--feature-pools", nargs="+", default=["cls", "mean"], choices=["cls", "mean"])
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--normalization", default="repo", choices=["repo", "imagenet", "clip"])
    parser.add_argument("--results-root", default="results/layer_probe")
    parser.add_argument("--raw-run-name", default="default")
    parser.add_argument("--trained-run-name", default="seed1994_final")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--wncm-reg", type=float, default=1e-4)
    parser.add_argument("--cov-max-samples", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true", help="Launch each worker detached with nohup-style subprocesses.")
    parser.add_argument("--keep-run-logs", action="store_true", help="Keep worker logs and manifests after successful foreground runs.")
    return parser.parse_args()


def absolute_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def protocol_result(
    output_dir: Path,
    dataset: str,
    backbone: str,
    run_name: str | None,
    normalization: str,
    method: str,
    protocol: str,
) -> Path:
    return (
        output_dir / "analyses" / "cross_factor" / method / dataset / safe_name(backbone)
        / safe_name(run_name) / normalization / protocol / "metrics.json"
    )


def features_exist(output_dir: Path, dataset: str, backbone: str, run_name: str | None, normalization: str) -> bool:
    path = output_dir / "features" / dataset / safe_name(backbone)
    if run_name:
        path = path / safe_name(run_name)
    path = path / normalization
    return any(path.glob("train_domain*.npz")) and any(path.glob("test_domain*.npz"))


def command_for_task(args, root: Path, source: str, dataset: str, alias: str, protocols: list[str]) -> tuple[list[str], dict]:
    results_root = absolute_path(root, args.results_root)
    output_dir = results_root / source
    run_name = args.trained_run_name if source == "slca_trained" else args.raw_run_name
    backbone = BACKBONE_NAMES[source][alias]
    command = [
        "conda", "run", "-n", "DGIL", "python", "scripts/eval_cross_factor_probes.py",
        "--method", args.method,
        "--dataset", dataset,
        "--backbone", alias,
        "--output-dir", str(output_dir),
        "--normalization", args.normalization,
        "--feature-source", source,
        "--protocols", *protocols,
        "--feature-pools", *args.feature_pools,
        "--layers", *[str(layer) for layer in args.layers],
        "--seed", str(args.seed),
        "--wncm-reg", str(args.wncm_reg),
        "--cov-max-samples", str(args.cov_max_samples),
    ]
    if args.keep_run_logs:
        command.append("--verbose-progress")
    if run_name:
        command.extend(["--run-name", run_name])
    if args.overwrite:
        command.append("--overwrite")
    task = {
        "feature_source": source,
        "dataset": dataset,
        "alias": alias,
        "backbone": backbone,
        "output_dir": str(output_dir),
        "run_name": run_name,
        "protocols": protocols,
        "command": command,
    }
    return command, task


def main():
    args = parse_args()
    root = project_root()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    manifest_root = absolute_path(root, args.results_root) / "runs" / "cross_factor" / args.method / timestamp
    manifest_root.mkdir(parents=True, exist_ok=True)
    tasks, skipped, missing_features = [], [], []

    results_root = absolute_path(root, args.results_root)
    for source in args.feature_sources:
        output_dir = results_root / source
        run_name = args.trained_run_name if source == "slca_trained" else args.raw_run_name
        for dataset in args.datasets:
            for alias in args.backbones:
                backbone = BACKBONE_NAMES[source][alias]
                if not features_exist(output_dir, dataset, backbone, run_name, args.normalization):
                    missing_features.append({"feature_source": source, "dataset": dataset, "alias": alias, "path": str(output_dir)})
                    continue
                pending_protocols = []
                for protocol in args.protocols:
                    path = protocol_result(output_dir, dataset, backbone, run_name, args.normalization, args.method, protocol)
                    if path.exists() and not args.overwrite:
                        skipped.append({"feature_source": source, "dataset": dataset, "alias": alias, "protocol": protocol, "result": str(path)})
                    else:
                        pending_protocols.append(protocol)
                if not pending_protocols:
                    continue
                _, task = command_for_task(args, root, source, dataset, alias, pending_protocols)
                tasks.append(task)

    worker_count = max(1, min(args.workers, len(tasks))) if tasks else 0
    queues = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        queues[index % worker_count].append(task)

    workers = []
    foreground_processes = []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    for worker_index, queue in enumerate(queues):
        log_path = manifest_root / f"worker_{worker_index:02d}.log"
        worker_manifest = manifest_root / f"worker_{worker_index:02d}.json"
        worker_manifest.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
        shell_commands = []
        for task in queue:
            quoted = subprocess.list2cmdline(task["command"])
            shell_commands.append(
                f"echo '[task-start] source={task['feature_source']} dataset={task['dataset']} alias={task['alias']} time='$(date); "
                f"{quoted}; "
                f"echo '[task-done] source={task['feature_source']} dataset={task['dataset']} alias={task['alias']} time='$(date)"
            )
        command = ["bash", "-lc", "set -euo pipefail; " + "; ".join(shell_commands)]
        worker_info = {"worker": worker_index, "log": str(log_path), "tasks": len(queue), "pid": None}
        if not args.dry_run:
            log_handle = log_path.open("w")
            if args.background:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
                worker_info["pid"] = process.pid
            else:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                worker_info["pid"] = process.pid
                foreground_processes.append((process, worker_info))
            log_handle.close()
        workers.append(worker_info)
        print(f"[worker] index={worker_index} tasks={len(queue)} log={log_path} pid={worker_info['pid']}", flush=True)

    for process, worker_info in foreground_processes:
        worker_info["returncode"] = process.wait()

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "tasks": tasks,
        "skipped": skipped,
        "missing_features": missing_features,
        "workers": workers,
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[done] tasks={len(tasks)} skipped={len(skipped)} missing_features={len(missing_features)} manifest={manifest_path}", flush=True)
    if missing_features:
        print("[warning] some requested feature caches are missing; see manifest", flush=True)
    failed = not args.background and any(worker.get("returncode", 0) != 0 for worker in workers)
    if failed:
        raise SystemExit("One or more workers failed; inspect worker logs")
    if not args.background and not args.keep_run_logs and not args.dry_run:
        shutil.rmtree(manifest_root)
        parent = manifest_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    main()
