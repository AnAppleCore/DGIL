import argparse
import ast
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path


RESULT_COLUMNS = [
    "model_name",
    "prefix",
    "backbone_type",
    "seed",
    "dataset",
    "average_accuracy_all",
    "last_accuracy_all",
    "last_accuracy_in",
    "last_accuracy_out",
    "last_accuracy_out_worst",
    "forgetting",
]

METRIC_COLUMNS = [
    "average_accuracy_all",
    "last_accuracy_all",
    "last_accuracy_in",
    "last_accuracy_out",
    "last_accuracy_out_worst",
    "forgetting",
]


def normalize_logged_value(value):
    value = re.sub(r"(?:np|numpy)\.float\d*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", r"\1", value)
    value = re.sub(r"(?:np|numpy)\.int\d*\(([-+]?\d+)\)", r"\1", value)
    return value


def safe_literal_eval(value, log_file, field_name):
    normalized_value = normalize_logged_value(value)
    try:
        return ast.literal_eval(normalized_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"failed to parse {field_name} in {log_file}: {value}") from exc


def normalize_backbone_type_for_results(backbone_type):
    method_suffixes = [
        "_coda_prompt",
        "_dualprompt",
        "_dot_l2p",
        "_sprompt",
        "_adapter",
        "_l2p",
        "_lae",
        "_dot",
    ]
    normalized = backbone_type
    changed = True
    while changed:
        changed = False
        for suffix in method_suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
                break

    ranpac_backbone_aliases = {
        "pretrained_vit_b16_224": "vit_base_patch16_224",
        "pretrained_vit_b16_224_clip": "vit_base_patch16_224_clip",
        "pretrained_vit_b16_224_mae": "vit_base_patch16_224_mae",
        "pretrained_vit_b16_224_21k_ibot": "vit_base_patch16_224_21k_ibot",
        "pretrained_vit_b14_224_dinov2": "vit_base_patch14_224_dinov2",
    }
    return ranpac_backbone_aliases.get(normalized, normalized)


def sample_std(values):
    if len(values) < 2:
        return ""
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_log_file(log_file, eval_key="CNN"):
    result_dict = {
        "model_name": "",
        "dataset": "",
        "prefix": "",
        "backbone_type": "",
        "seed": "",
        "average_accuracy_all": 0.0,
        "last_accuracy_all": 0.0,
        "last_accuracy_in": 0.0,
        "last_accuracy_out": 0.0,
        "last_accuracy_out_worst": 0.0,
        "forgetting": -1e-5,
    }

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    task_reference_domains = {}
    class_id_pairs = []
    domain_accuracies = {}
    no_nme = False

    for line in lines:
        if "No NME" in line and eval_key == "NME":
            no_nme = True
        if "[trainer.py] => model_name:" in line:
            result_dict["model_name"] = line.split(":")[-1].strip()
        if "[trainer.py] => prefix" in line:
            result_dict["prefix"] = line.split(":")[-1].strip()
        if "[trainer.py] => backbone_type:" in line:
            result_dict["backbone_type"] = normalize_backbone_type_for_results(line.split(":")[-1].strip())
        if "[trainer.py] => seed:" in line:
            result_dict["seed"] = line.split(":")[-1].strip()
        if "[trainer.py] => dataset:" in line:
            result_dict["dataset"] = line.split(":")[-1].strip()
        if f"[trainer.py] => Average Accuracy ({eval_key}):" in line:
            result_dict["average_accuracy_all"] = float(line.split(":")[-1].strip())
        if f"[trainer.py] => Last Accuracy ({eval_key}):" in line:
            result_dict["last_accuracy_all"] = float(line.split(":")[-1].strip())
        if f"[trainer.py] => {eval_key}: {{'total':" in line:
            acc_dict = safe_literal_eval(line.split(f"{eval_key}:")[-1].strip(), log_file, eval_key)
            result_dict["last_accuracy_all"] = acc_dict.get("total", 0.0)
        if f"[trainer.py] => Forgetting ({eval_key}):" in line:
            result_dict["forgetting"] = float(line.split(":")[-1].strip())
        if "=> Task" in line and "reference domain is" in line:
            task_id = int(line.split("Task")[1].split(":")[0].strip())
            domain_part = line.split("reference domain is")[-1].strip()
            domain_id = int(domain_part.split("[")[1].split("]")[0].strip())
            task_reference_domains[task_id] = domain_id
        if "[trainer.py] => Class ID pairs:" in line:
            class_id_pairs = safe_literal_eval(
                line.split("[trainer.py] => Class ID pairs:")[-1].strip(), log_file, "Class ID pairs"
            )
        if "[trainer.py] => Domain [" in line and f"{eval_key}:" in line:
            domain_part = line.split("[trainer.py] => Domain")[-1].strip()
            domain_id = int(domain_part.split("[")[1].split("]")[0].strip())
            accuracies = safe_literal_eval(
                line.split(f"{eval_key}:")[-1].strip(), log_file, f"Domain {domain_id} {eval_key}"
            )
            domain_accuracies[domain_id] = accuracies

    if no_nme:
        return result_dict, False, "No NME"
    if result_dict["forgetting"] < 0.0:
        return result_dict, False, "missing forgetting"

    in_accuracies = []
    out_accuracies = []
    out_worst_accuracies = []

    for task_id, ref_domain in task_reference_domains.items():
        if task_id >= len(class_id_pairs) or ref_domain not in domain_accuracies:
            continue

        class_pair = class_id_pairs[task_id]
        class_key = f"{class_pair[0]:02d}-{class_pair[1]:02d}"

        in_accuracies.append(domain_accuracies[ref_domain].get(class_key, 0.0))

        out_domains = [domain for domain in domain_accuracies if domain != ref_domain]
        out_accs = [domain_accuracies[domain].get(class_key, 0.0) for domain in out_domains]
        if out_accs:
            out_accuracies.append(sum(out_accs) / len(out_accs))
            out_worst_accuracies.append(min(out_accs))

    result_dict["last_accuracy_in"] = sum(in_accuracies) / len(in_accuracies) if in_accuracies else 0.0
    result_dict["last_accuracy_out"] = sum(out_accuracies) / len(out_accuracies) if out_accuracies else 0.0
    result_dict["last_accuracy_out_worst"] = (
        sum(out_worst_accuracies) / len(out_worst_accuracies) if out_worst_accuracies else 0.0
    )

    return result_dict, True, ""


def build_incomplete_info(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["prefix"], row["backbone_type"], row["dataset"])].append(str(row["seed"]))

    incomplete_info = []
    for (model_name, prefix, backbone_type, dataset), seeds in sorted(grouped.items()):
        existing_seeds = sorted(seeds)
        incomplete_info.append(
            {
                "model_name": model_name,
                "prefix": prefix,
                "backbone_type": backbone_type,
                "dataset": dataset,
                "seed_count": len(existing_seeds),
                "existing_seeds": existing_seeds,
            }
        )
    return incomplete_info


def build_mean_std(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["prefix"], row["backbone_type"], row["dataset"])].append(row)

    mean_std_rows = []
    for (model_name, prefix, backbone_type, dataset), group_rows in sorted(grouped.items()):
        result = {
            "model_name_": model_name,
            "prefix_": prefix,
            "backbone_type_": backbone_type,
            "dataset_": dataset,
        }
        for column in METRIC_COLUMNS:
            values = [float(row[column]) for row in group_rows]
            result[f"{column}_mean"] = sum(values) / len(values)
            result[f"{column}_std"] = sample_std(values)
        mean_std_rows.append(result)
    return mean_std_rows


def parse_log_folder(log_folder, eval_key="CNN", output_dir=None, valid_seeds=None, include_screen_logs=False):
    log_folder = Path(log_folder).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else log_folder / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_result = []
    invalid_logs = []

    for root, dirs, files in os.walk(log_folder):
        if not include_screen_logs:
            dirs[:] = [directory for directory in dirs if directory != "screen_logs"]

        for file in files:
            if not file.endswith(".log"):
                continue

            log_file = Path(root) / file
            try:
                result_dict, is_valid, reason = parse_log_file(log_file, eval_key)
            except ValueError as exc:
                invalid_logs.append({"log_file": str(log_file), "reason": str(exc)})
                continue

            if is_valid:
                all_result.append(result_dict)
            else:
                invalid_logs.append({"log_file": str(log_file), "reason": reason})

    all_result.sort(key=lambda row: (row["model_name"], row["dataset"], row["prefix"], row["backbone_type"], str(row["seed"])))
    write_csv(output_dir / f"{eval_key}_results.csv", all_result, RESULT_COLUMNS)
    write_csv(output_dir / f"{eval_key}_invalid_logs.csv", invalid_logs, ["log_file", "reason"])

    if not all_result:
        write_csv(output_dir / f"{eval_key}_results_filtered.csv", [], RESULT_COLUMNS)
        write_csv(
            output_dir / f"{eval_key}_incomplete_experiments.csv",
            [],
            ["model_name", "prefix", "backbone_type", "dataset", "seed_count", "existing_seeds"],
        )
        write_csv(output_dir / f"{eval_key}_results_mean_std_filtered.csv", [], ["model_name_", "prefix_", "backbone_type_", "dataset_"])
        print(f"No valid {eval_key} logs found under {log_folder}")
        print(f"Wrote result files to {output_dir}")
        return all_result

    valid_seeds = [str(seed) for seed in (valid_seeds or ["1994", "1995", "1996"])]
    filtered_rows = [row for row in all_result if str(row["seed"]) in valid_seeds]

    grouped = defaultdict(list)
    for row in filtered_rows:
        grouped[(row["model_name"], row["prefix"], row["backbone_type"], row["dataset"])].append(row)

    complete_rows = []
    incomplete_rows = []
    for group_rows in grouped.values():
        seed_count = len({str(row["seed"]) for row in group_rows})
        if seed_count == len(valid_seeds):
            complete_rows.extend(group_rows)
        else:
            incomplete_rows.extend(group_rows)

    complete_rows.sort(key=lambda row: (row["model_name"], row["dataset"], row["prefix"], row["backbone_type"], str(row["seed"])))
    write_csv(output_dir / f"{eval_key}_results_filtered.csv", complete_rows, RESULT_COLUMNS)

    incomplete_info = build_incomplete_info(incomplete_rows)
    write_csv(
        output_dir / f"{eval_key}_incomplete_experiments.csv",
        incomplete_info,
        ["model_name", "prefix", "backbone_type", "dataset", "seed_count", "existing_seeds"],
    )

    mean_std_rows = build_mean_std(complete_rows)
    mean_std_columns = ["model_name_", "prefix_", "backbone_type_", "dataset_"]
    for column in METRIC_COLUMNS:
        mean_std_columns.extend([f"{column}_mean", f"{column}_std"])
    write_csv(output_dir / f"{eval_key}_results_mean_std_filtered.csv", mean_std_rows, mean_std_columns)

    if incomplete_info:
        print(f"Found {len(incomplete_info)} incomplete experiment groups for {eval_key}.")
    else:
        print(f"All experiment groups have complete seeds ({', '.join(valid_seeds)}) for {eval_key}.")
    print(f"Parsed {len(all_result)} valid {eval_key} logs under {log_folder}.")
    print(f"Skipped {len(invalid_logs)} invalid or incomplete {eval_key} logs.")
    print(f"Wrote result files to {output_dir}")
    return all_result


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DGIL experiment results from trainer log files.")
    parser.add_argument("--root", default=".", help="Root folder to recursively scan for .log files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated CSV files. Defaults to <root>/logs.")
    parser.add_argument("--eval-key", nargs="+", default=["CNN"], help="Evaluation head(s) to parse, e.g. CNN NME.")
    parser.add_argument("--seeds", nargs="+", default=["1994", "1995", "1996"], help="Seeds required for complete groups.")
    parser.add_argument("--include-screen-logs", action="store_true", help="Also scan logs/screen_logs console output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    root_folder = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_folder / "logs"

    for eval_key in args.eval_key:
        parse_log_folder(
            root_folder,
            eval_key=eval_key,
            output_dir=output_dir,
            valid_seeds=args.seeds,
            include_screen_logs=args.include_screen_logs,
        )


if __name__ == "__main__":
    main()
