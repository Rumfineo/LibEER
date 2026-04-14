import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_PATTERN = re.compile(
    r"ALLRound Mean and Std of (?P<metric>[^:]+) : (?P<mean>\d+(?:\.\d+)?)/(?P<std>\d+(?:\.\d+)?)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LibEER DGCNN sequentially with multiple seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Seed list.")
    parser.add_argument(
        "--python-exe", default=sys.executable, help="Python executable used to launch training."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0"],
        help="Training device passed to DGCNN_train.py.",
    )
    parser.add_argument(
        "--base-log-dir",
        default=None,
        help="Base directory for per-seed logs. Defaults to <repo>/log/dgcnn_multi_seed.",
    )
    parser.add_argument(
        "--base-output-dir",
        default=None,
        help="Base directory for per-seed outputs. Defaults to <repo>/result/dgcnn_multi_seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to DGCNN_train.py. Put them after --.",
    )
    return parser.parse_args()


def strip_separator(args):
    if args and args[0] == "--":
        return args[1:]
    return args


def normalize_train_args(args):
    normalized = []
    for arg in args:
        if arg.startswith("--") and len(arg) > 2:
            normalized.append(f"-{arg[2:]}")
        else:
            normalized.append(arg)
    return normalized


def default_train_args(dataset_path):
    return [
        "-dataset",
        "seed_de_lds",
        "-dataset_path",
        str(dataset_path),
        "-setting",
        "seed_sub_dependent_front_back_setting",
        "-metrics",
        "acc",
        "macro-f1",
        "-metric_choose",
        "acc",
        "-onehot",
        "-num_workers",
        "0",
    ]


def parse_summary(log_path):
    summary = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = SUMMARY_PATTERN.search(line)
        if match:
            summary[match.group("metric")] = {
                "mean": float(match.group("mean")),
                "std": float(match.group("std")),
            }
    return summary


def write_summary(summary_path, rows):
    metrics = sorted({metric for row in rows for metric in row["summary"].keys()})
    fieldnames = ["seed", "status"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = {"seed": row["seed"], "status": row["status"]}
            for metric in metrics:
                values = row["summary"].get(metric, {})
                item[f"{metric}_mean"] = values.get("mean", "")
                item[f"{metric}_std"] = values.get("std", "")
            writer.writerow(item)


def safe_write(line):
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
    sys.stdout.flush()


def main():
    args = parse_args()
    extra_train_args = normalize_train_args(strip_separator(args.train_args))

    repo_root = Path(__file__).resolve().parents[1]
    workdir = repo_root / "LibEER"
    train_script = workdir / "DGCNN_train.py"
    dataset_path = workdir / "Datasets" / "SEED"
    base_log_dir = Path(args.base_log_dir) if args.base_log_dir else repo_root / "log" / "dgcnn_multi_seed"
    base_output_dir = (
        Path(args.base_output_dir) if args.base_output_dir else repo_root / "result" / "dgcnn_multi_seed"
    )
    base_log_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = 0
    for seed in args.seeds:
        seed_name = f"seed_{seed}"
        seed_log_dir = base_log_dir / seed_name
        seed_output_dir = base_output_dir / seed_name
        metrics_dir = seed_log_dir / "metrics"
        seed_log_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        run_log = seed_log_dir / "run.log"

        command = [
            args.python_exe,
            str(train_script),
            *default_train_args(dataset_path),
            *extra_train_args,
            "-seed",
            str(seed),
            "-device",
            args.device,
            "-log_dir",
            str(metrics_dir),
            "-output_dir",
            str(seed_output_dir),
        ]

        print(f"\n=== Running seed {seed} ===")
        print(" ".join(command))
        if args.dry_run:
            results.append({"seed": seed, "status": "dry-run", "summary": {}})
            continue

        with run_log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                safe_write(line)
                handle.write(line)
            return_code = process.wait()

        summary = parse_summary(run_log)
        status = "ok" if return_code == 0 else f"failed({return_code})"
        if return_code != 0:
            failures += 1
        results.append({"seed": seed, "status": status, "summary": summary})

        summary_json = seed_log_dir / "summary.json"
        summary_json.write_text(
            json.dumps({"seed": seed, "status": status, "summary": summary}, indent=2),
            encoding="utf-8",
        )

        write_summary(base_log_dir / "summary.csv", results)

    print("\n=== Summary ===")
    for row in results:
        print(json.dumps(row, ensure_ascii=False))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
