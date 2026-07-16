#!/usr/bin/env python3

import csv
import subprocess
from pathlib import Path

ROOT_DIR = Path("/common/home/rvk22/spaceforge-xai-run2")
TRACKER_CSV = ROOT_DIR / "SpaceForge-xai Job tracker - Analytics.csv"
SLURM_FILE = ROOT_DIR / "Sim" / "run_orbit3.slurm"

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
            config_name = row.get("Config", "").strip()

            if not config_name:
                continue

            # Keep exact CSV order, but avoid submitting duplicate names twice.
            if config_name not in config_names:
                config_names.append(config_name)

    print("Configs found:")
    for name in config_names:
        print(f"  {name}")

    print()
    print(f"Submitting {len(config_names)} configs...")

    for config_name in config_names:
        cmd = [
            "sbatch",
            str(SLURM_FILE),
            config_name,
        ]

        print(" ".join(cmd))
        result = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True)

        if result.returncode != 0:
            print(f"WARNING: failed to submit {config_name}")

if __name__ == "__main__":
    main()