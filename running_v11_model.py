# -*- coding: utf-8 -*-
"""Colab launcher for SpaceForge Graph WaveNet v11.

Upload this file and train_v11.py to Colab, or place train_v11.py at:
    MyDrive/SpaceForgeData/scripts/train_v11.py

Run in Colab with:
    %run /content/running_v11_model.py

The launcher stores every run in a unique Drive directory, streams a live
console log to Drive, periodically snapshots local outputs, and creates flat
CSV reports for per-seed results and seed-averaged mean/std metrics.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive


# -----------------------------------------------------------------------------
# Colab runner configuration
# -----------------------------------------------------------------------------
MOUNT = "/content/gdrive"
FORCE_REMOUNT = False
CLEAR_LOCAL_WORK = True
SNAPSHOT_INTERVAL_SECONDS = 180
MINIMUM_SEED_COUNT = 3

CLEAN_ROOT = Path(MOUNT) / "MyDrive/SpaceForgeData/spaceforge-cleaned2/sf-cleaned-2"
RESULTS_ROOT = Path(MOUNT) / "MyDrive/SpaceForgeData/spaceforge-cleaned2/graphwavenet_xai_outputs_v11"
DRIVE_SCRIPT = Path(MOUNT) / "MyDrive/SpaceForgeData/scripts/train_v11.py"
LOCAL_UPLOADED_SCRIPT = Path("/content/train_v11.py")
LOCAL_RUNTIME_SCRIPT = Path("/content/train_v11_runtime.py")
LOCAL_WORK = Path("/content/spaceforge_graphwavenet_work_v11")
LOCAL_OUTPUT = LOCAL_WORK / "outputs"

CHUNKS_DIR = CLEAN_ROOT / "all_runs_features_model_ready_chunks"
LABEL_PATH = CLEAN_ROOT / "streaming_funnel_outputs" / "run_funnel_df.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_training_script() -> Path:
    candidates = [
        DRIVE_SCRIPT,
        LOCAL_UPLOADED_SCRIPT,
        Path(__file__).resolve().with_name("train_v11.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"train_v11.py was not found. Checked:\n{checked}")


def read_literal_cfg(source_text: str) -> dict:
    tree = ast.parse(source_text)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "CFG" for target in node.targets):
            cfg = ast.literal_eval(node.value)
            if not isinstance(cfg, dict):
                break
            return cfg
    raise ValueError("Could not read the literal CFG dictionary from train_v11.py")


def create_run_directory() -> tuple[str, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = base
    run_dir = RESULTS_ROOT / "runs" / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{base}_{suffix:02d}"
        run_dir = RESULTS_ROOT / "runs" / run_id
        suffix += 1
    run_dir.mkdir(parents=True)
    atomic_write_text(RESULTS_ROOT / "latest_run_path.txt", str(run_dir))
    return run_id, run_dir


def patch_runtime_source(source_text: str, run_dir: Path) -> str:
    text = source_text.replace("/content/drive/MyDrive", "/content/gdrive/MyDrive")

    root_pattern = re.compile(
        r'^(\s*["\']drive_root["\']\s*:\s*)["\'][^"\']*["\'](\s*,)\s*$',
        flags=re.MULTILINE,
    )
    text, root_count = root_pattern.subn(
        rf'\1{str(CLEAN_ROOT)!r}\2',
        text,
        count=1,
    )

    cfg_output_pattern = re.compile(
        r'^(\s*["\']drive_output_root["\']\s*:\s*)["\'][^"\']*["\'](\s*,)\s*$',
        flags=re.MULTILINE,
    )
    text, cfg_output_count = cfg_output_pattern.subn(
        rf'\1{str(run_dir)!r}\2',
        text,
        count=1,
    )

    output_pattern = re.compile(r'^DRIVE_OUTPUT\s*=.*$', flags=re.MULTILINE)
    replacement = f"DRIVE_OUTPUT = Path({str(run_dir)!r})"
    text, output_count = output_pattern.subn(replacement, text, count=1)
    if root_count != 1 or cfg_output_count != 1 or output_count != 1:
        raise ValueError(
            "Expected exactly one v11 drive_root, drive_output_root, and DRIVE_OUTPUT "
            "assignment; refusing to run because paths might be wrong."
        )
    return text


def validate_inputs(cfg: dict) -> list[int]:
    chunk_count = len(list(CHUNKS_DIR.glob("part_*.parquet"))) if CHUNKS_DIR.exists() else 0
    seeds = [int(seed) for seed in cfg.get("seeds", [cfg.get("seed", 42)])]

    print("chunks directory:", CHUNKS_DIR, "exists:", CHUNKS_DIR.exists())
    print("chunk files:", chunk_count)
    print("labels:", LABEL_PATH, "exists:", LABEL_PATH.exists())
    print("configured seeds:", seeds)
    print("run baselines:", bool(cfg.get("run_baselines", False)))

    if chunk_count == 0:
        raise FileNotFoundError(f"No part_*.parquet files found in {CHUNKS_DIR}")
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed list contains duplicates: {seeds}")
    if len(seeds) < MINIMUM_SEED_COUNT:
        raise ValueError(
            f"v11 seed averaging requires at least {MINIMUM_SEED_COUNT} seeds; got {seeds}"
        )
    return seeds


def safe_clear_local_work() -> None:
    expected = Path("/content/spaceforge_graphwavenet_work_v11")
    if LOCAL_WORK != expected or not str(LOCAL_WORK).startswith("/content/spaceforge_"):
        raise RuntimeError(f"Refusing to clear unexpected path: {LOCAL_WORK}")
    if LOCAL_WORK.exists():
        shutil.rmtree(LOCAL_WORK)
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def snapshot_outputs(src_root: Path, dst_root: Path) -> dict:
    copied = 0
    skipped = 0
    bytes_copied = 0
    if not src_root.exists():
        return {"copied": 0, "skipped": 0, "bytes_copied": 0}

    for src in src_root.rglob("*"):
        if not src.is_file() or src.name.endswith(".partial"):
            continue
        dst = dst_root / src.relative_to(src_root)
        try:
            unchanged = dst.exists() and dst.stat().st_size == src.stat().st_size
        except OSError:
            unchanged = False
        if unchanged:
            skipped += 1
            continue
        copy_file_atomic(src, dst)
        copied += 1
        bytes_copied += src.stat().st_size
    return {"copied": copied, "skipped": skipped, "bytes_copied": bytes_copied}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def create_seed_reports(run_dir: Path) -> tuple[Path, Path]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Seed report source is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    multi_seed = summary.get("multi_seed", {})

    average_rows = []
    for model, metrics in multi_seed.get("aggregated", {}).items():
        for metric, stats in metrics.items():
            average_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "n_seeds": multi_seed.get("n_seeds"),
                }
            )

    per_seed_rows = []
    for model, records in multi_seed.get("per_seed", {}).items():
        for record in records:
            per_seed_rows.append({"model": model, **record})

    averages_path = run_dir / "seed_averaged_metrics.csv"
    per_seed_path = run_dir / "per_seed_metrics.csv"
    write_csv(averages_path, average_rows)
    write_csv(per_seed_path, per_seed_rows)

    if not average_rows or not per_seed_rows:
        raise ValueError("run_summary.json did not contain complete multi-seed results")
    return averages_path, per_seed_path


def create_manifest(run_dir: Path, metadata: dict) -> Path:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(".partial"):
            files.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "bytes": path.stat().st_size,
                }
            )
    manifest_path = run_dir / "run_manifest.json"
    atomic_write_json(manifest_path, {**metadata, "files": files})
    return manifest_path


def print_saved_results(run_dir: Path) -> None:
    print("\nSaved result files:")
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(".partial"):
            size_mb = path.stat().st_size / 1_000_000
            print(f"  {path.relative_to(run_dir)} | {size_mb:.3f} MB")


def main() -> None:
    print("Mounting Google Drive at", MOUNT)
    drive.mount(MOUNT, force_remount=FORCE_REMOUNT)

    source_path = find_training_script()
    source_text = source_path.read_text(encoding="utf-8", errors="strict")
    cfg = read_literal_cfg(source_text)
    seeds = validate_inputs(cfg)
    run_id, run_dir = create_run_directory()

    print("training script:", source_path)
    print("run id:", run_id)
    print("Drive run directory:", run_dir)

    runtime_text = patch_runtime_source(source_text, run_dir)
    LOCAL_RUNTIME_SCRIPT.write_text(runtime_text, encoding="utf-8")

    # Keep both original and exact runtime sources with the results for provenance.
    copy_file_atomic(source_path, run_dir / "train_v11_source.py")
    copy_file_atomic(LOCAL_RUNTIME_SCRIPT, run_dir / "train_v11_runtime.py")

    metadata = {
        "run_id": run_id,
        "started_utc": utc_now(),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "runtime_sha256": sha256_file(LOCAL_RUNTIME_SCRIPT),
        "seeds": seeds,
        "seed_count": len(seeds),
        "run_baselines": bool(cfg.get("run_baselines", False)),
        "status": "starting",
    }
    status_path = run_dir / "colab_run_status.json"
    atomic_write_json(status_path, metadata)

    if CLEAR_LOCAL_WORK:
        safe_clear_local_work()
    else:
        LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-u", str(LOCAL_RUNTIME_SCRIPT)]
    live_log_path = run_dir / "colab_live_console.log"
    next_snapshot = time.monotonic() + SNAPSHOT_INTERVAL_SECONDS
    process = None

    print("starting:", " ".join(command))
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        metadata.update({"status": "running", "pid": process.pid})
        atomic_write_json(status_path, metadata)

        with live_log_path.open("a", encoding="utf-8", buffering=1) as live_log:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                live_log.write(line)

                if time.monotonic() >= next_snapshot:
                    try:
                        snap = snapshot_outputs(LOCAL_OUTPUT, run_dir)
                        metadata.update(
                            {
                                "status": "running",
                                "last_snapshot_utc": utc_now(),
                                "last_snapshot": snap,
                            }
                        )
                        atomic_write_json(status_path, metadata)
                        print(f"[runner] Drive snapshot: {snap}", flush=True)
                    except Exception as exc:
                        print(f"[runner] WARNING: snapshot failed: {exc}", flush=True)
                    next_snapshot = time.monotonic() + SNAPSHOT_INTERVAL_SECONDS

        return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        snapshot_outputs(LOCAL_OUTPUT, run_dir)
        metadata.update({"status": "interrupted", "finished_utc": utc_now()})
        atomic_write_json(status_path, metadata)
        raise
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
        snapshot_outputs(LOCAL_OUTPUT, run_dir)
        metadata.update({"status": "runner_error", "finished_utc": utc_now()})
        atomic_write_json(status_path, metadata)
        raise

    final_snapshot = snapshot_outputs(LOCAL_OUTPUT, run_dir)
    metadata.update(
        {
            "status": "complete" if return_code == 0 else "training_failed",
            "return_code": return_code,
            "finished_utc": utc_now(),
            "final_snapshot": final_snapshot,
        }
    )
    atomic_write_json(status_path, metadata)

    if return_code != 0:
        create_manifest(run_dir, metadata)
        print_saved_results(run_dir)
        raise RuntimeError(
            f"train_v11.py exited with code {return_code}. Results and logs were preserved in {run_dir}"
        )

    averages_path, per_seed_path = create_seed_reports(run_dir)
    manifest_path = create_manifest(run_dir, metadata)
    atomic_write_text(RESULTS_ROOT / "latest_successful_run_path.txt", str(run_dir))

    print_saved_results(run_dir)
    print("\nTraining complete.")
    print("results:", run_dir)
    print("seed averages:", averages_path)
    print("per-seed metrics:", per_seed_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
