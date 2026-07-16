#!/usr/bin/env python3
# python3 scripts/submit_all_configs_serial.py

import csv
import re
import subprocess
from pathlib import Path


ROOT_DIR = Path("/common/home/rvk22/spaceforge-xai-run2")
TRACKER_CSV = ROOT_DIR / "SpaceForge-xai Job tracker - Analytics.csv"
SLURM_FILE = ROOT_DIR / "Sim" / "run_orbit3.slurm"


def config_sort_key(name):
    """
    Sort configs in a human readable order.

    Keeps A-Config rows first, then Config rows, then anything else.
    """
    text = name.strip()

    match_a = re.match(r"^A-Config\s*(\d+)$", text, flags=re.IGNORECASE)
    if match_a:
        return (0, int(match_a.group(1)), text)

    match_regular = re.match(r"^Config\s*(\d+)$", text, flags=re.IGNORECASE)
    if match_regular:
        return (1, int(match_regular.group(1)), text)

    return (2, 999999, text)


def parse_job_id(sbatch_output):
    """
    Extract the numeric Slurm job id from text like:
    Submitted batch job 132784
    """
    match = re.search(r"Submitted batch job\s+(\d+)", sbatch_output)

    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {sbatch_output}")

    return match.group(1)


def main():
    if not TRACKER_CSV.exists():
        raise FileNotFoundError(f"Tracker CSV not found: {TRACKER_CSV}")

    if not SLURM_FILE.exists():
        raise FileNotFoundError(f"Slurm file not found: {SLURM_FILE}")

    config_names = []

    with TRACKER_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if "Config" not in reader.fieldnames:
            raise RuntimeError("CSV must contain a column named Config")

        for row in reader:
            name = (row.get("Config") or "").strip()

            if not name:
                continue

            if name not in config_names:
                config_names.append(name)

    config_names = sorted(config_names, key=config_sort_key)

    print("Configs to submit in serial order:")
    for name in config_names:
        print(f"  {name}")

    previous_job_id = None

    for config_name in config_names:
        cmd = [
            "sbatch",
        ]

        if previous_job_id is not None:
            cmd.append(f"--dependency=afterany:{previous_job_id}")

        cmd.extend(
            [
                str(SLURM_FILE),
                config_name,
            ]
        )

        print()
        print("Submitting:", " ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            capture_output=True,
        )

        print(result.stdout.strip())

        if result.stderr.strip():
            print(result.stderr.strip())

        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit config {config_name}")

        previous_job_id = parse_job_id(result.stdout)

    print()
    print("Done.")
    print(f"Last submitted job id: {previous_job_id}")


if __name__ == "__main__":
    main()