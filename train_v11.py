"""
SpaceForge Graph WaveNet v11 -- dual-range TTF health-monitoring script.

This complete v11 successor combines v8's all-range exact-TTF supervision with
v10's dense near-failure timing targets. It adds global and near-refinement TTF
heads, run-balanced model selection, and native graph/feature support for the
ArrayGimbal, BatteryThermal, CryoPanel, Radiator, and SourceInventory nodes.

Run on Colab:
    !python train_v11.py

All knobs live in CFG below. After each run the script writes:
  - train_v11_console_log.txt
  - run_summary.json
  - model_comparison.csv
  - training_history_{model}_{seed}.csv
  - per_class_cause_metrics.csv
  - per_horizon_metrics.csv
  - ttf_bin_metrics.csv
  - ttf_regression_metrics.csv
  - xai_* CSVs and PNG heatmaps
  - model_{model}_{seed}.pt

Everything is written locally then synced to:
  /content/gdrive/MyDrive/SpaceForgeData/spaceforge-cleaned2/graphwavenet_xai_outputs_v11
"""

# =============================================================================
# CFG -- all tunable knobs live here
# =============================================================================
CFG = {
    # data
    "drive_mount":       "/content/gdrive",
    "drive_root":        "/content/gdrive/MyDrive/SpaceForgeData/spaceforge-cleaned2/sf-cleaned-2",
    "drive_output_root": "/content/gdrive/MyDrive/SpaceForgeData/spaceforge-cleaned2/graphwavenet_xai_outputs_v11",
    "local_work_root":   "/content/spaceforge_graphwavenet_work_v11",

    # windowing
    "window_len":        64,
    "window_stride":     5,
    "near_failure_stride": 1,
    "near_failure_stride_window_ticks": 120.0,
    "ttf_norm_ticks":    240.0,
    "max_windows_per_split": None,   # set e.g. 2000 for a quick smoke test

    # A modest capacity increase accommodates 13 physical nodes while keeping
    # the six-layer, 64-tick receptive field unchanged.
    "residual_channels":  40,
    "dilation_channels":  40,
    "skip_channels":      80,
    "end_channels":       128,
    "kernel_size":        2,
    "blocks":             1,
    "layers":             6,
    "dropout":            0.18,
    "graph_order":        2,
    "adaptive_adj":       True,
    "node_embedding_dim": 16,
    "context_dropout":    0.40,

    # training
    "seed":               42,
    "batch_size":         128,
    "max_epochs":         40,
    "patience":           10,
    "learning_rate":      1e-3,
    "weight_decay":       1e-4,
    "cause_loss_weight":  1.00,
    "family_loss_weight": 0.50,
    "effusion_subtype_loss_weight": 0.50,
    "ttf_loss_weight":    1.00,
    "global_ttf_loss_weight": 0.75,
    "near_ttf_loss_weight":   1.25,
    "cause_specific_ttf_loss_weight": 0.35,
    "ttf_bin_loss_weight":0.60,
    "horizon_loss_weight":0.75,
    "monotonic_horizon_loss_weight": 0.10,
    "use_ttf":            True,

    # Dual-range timing supervision. The global head sees every positive
    # pre-failure window; the refinement head specializes inside 120 ticks.
    "cause_near_window_ticks": 120.0,
    "ttf_near_window_ticks":   60.0,
    "ttf_refine_window_ticks": 120.0,
    "near_ttf_loss_multiplier":2.00,
    "horizon_ticks":          [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0],
    "horizon_threshold":      0.50,

    # composite validation score
    "score_binary_auroc_weight": 1.00,
    "score_cause_f1_weight":     0.50,
    "score_horizon_f1_weight":   0.50,
    "score_ttf_bin_f1_weight":   0.50,
    "score_ttf_overall_mae_weight": 0.0020,
    "score_ttf_near_mae_weight":    0.0040,

    # explainer
    "explain_epochs":          150,
    "explain_lr":              0.05,
    "edge_size_penalty":       0.005,
    "edge_entropy_penalty":    0.10,
    "feature_size_penalty":    0.010,
    "feature_entropy_penalty": 0.10,
    "context_size_penalty":    0.010,
    "context_entropy_penalty": 0.10,
    "temporal_size_penalty":   0.005,
    "temporal_entropy_penalty":0.05,
    "temporal_smoothness_penalty": 0.05,

    # ablation / multi-seed
    "seeds":         [42, 123, 456],
    "run_baselines": True,

    # experiment log
    "run_tag": "R11-dual-range-ttf-five-node-expansion",

    # XAI
    "xai_model_keys": ["A", "B"],
    "xai_samples_per_cause": 10,
    "xai_mask_seeds": [11, 22, 33],
    "xai_target_kinds": ["binary", "flat_cause", "family", "horizon", "ttf_bin", "ttf_exact"],
    "xai_top_k_values": [1, 3, 5, 10],
    "xai_temporal_segments": [5, 10, 20, 30, 60],
}
# =============================================================================
# Imports
# =============================================================================
import sys
import subprocess
import os
import json
import math
import time
import random
import shutil
import warnings
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext
from datetime import datetime

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import pyarrow  # noqa
except Exception:
    pip_install("pyarrow")

try:
    import sklearn  # noqa
except Exception:
    pip_install("scikit-learn")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =============================================================================
# Helpers
# =============================================================================

_CONSOLE_LOG_BUFFER = []

def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    _CONSOLE_LOG_BUFFER.append(line)
    print(line, flush=True)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED   = CFG["seed"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED)
log(f"device: {DEVICE}")

# =============================================================================
# Paths
# =============================================================================
try:
    from google.colab import drive
    drive.mount(CFG.get("drive_mount", "/content/gdrive"))
except Exception as exc:
    log(f"Drive mount skipped: {exc}")

CLEAN_ROOT       = Path(CFG["drive_root"])
# Fallback for notebooks that were previously mounted at /content/drive.
if not CLEAN_ROOT.exists() and str(CLEAN_ROOT).startswith("/content/gdrive"):
    alt_root = Path(str(CLEAN_ROOT).replace("/content/gdrive", "/content/drive", 1))
    if alt_root.exists():
        log(f"drive_root fallback: {CLEAN_ROOT} not found, using {alt_root}")
        CLEAN_ROOT = alt_root

DRIVE_OUTPUT     = Path(CFG.get("drive_output_root") or (CLEAN_ROOT.parent / "graphwavenet_xai_outputs_v11"))
if not DRIVE_OUTPUT.exists() and str(DRIVE_OUTPUT).startswith("/content/gdrive"):
    alt_output = Path(str(DRIVE_OUTPUT).replace("/content/gdrive", "/content/drive", 1))
    if alt_output.parent.exists():
        log(f"drive_output_root fallback: {DRIVE_OUTPUT} not found, using {alt_output}")
        DRIVE_OUTPUT = alt_output
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

LOCAL_WORK       = Path(CFG["local_work_root"])
LOCAL_DATA       = LOCAL_WORK / "data"
LOCAL_OUTPUT     = LOCAL_WORK / "outputs"
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = LOCAL_OUTPUT

def read_latest_export_pointer(root: Path) -> Optional[Path]:
    for p in [root / "exports" / "latest_export_path.txt",
              CLEAN_ROOT.parent / "exports" / "latest_export_path.txt"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                q = Path(txt)
                if q.exists():
                    return q
    return None

LATEST_EXPORT = read_latest_export_pointer(CLEAN_ROOT)

# Streaming funnel label file (5,080 runs, matches the chunks) takes priority.
label_candidates = [
    CLEAN_ROOT / "streaming_funnel_outputs" / "run_funnel_df.csv",
]
if LATEST_EXPORT:
    label_candidates += [LATEST_EXPORT / "summaries" / "run_funnel_df.csv"]
label_candidates += [
    CLEAN_ROOT / "latest_export" / "summaries" / "run_funnel_df.csv",
    CLEAN_ROOT / "exports" / "latest_export" / "summaries" / "run_funnel_df.csv",
    CLEAN_ROOT / "run_funnel_df.csv",
    CLEAN_ROOT / "summaries" / "run_funnel_df.csv",
]

def first_existing(paths, label):
    for p in paths:
        if p.exists():
            log(f"{label} found: {p}")
            return p
    raise FileNotFoundError(f"No {label} path found. Checked:\n" + "\n".join(str(p) for p in paths))

def stage(src: Path) -> Path:
    dst = LOCAL_DATA / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        log(f"staging {src.name} to local disk …")
        shutil.copy2(src, dst)
    else:
        log(f"using cached local {src.name}")
    return dst

def stage_chunks(src_dir: Path) -> Path:
    """Merge chunked parquet directory into one local file with a unified schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks = sorted(src_dir.glob("part_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No part_*.parquet files in {src_dir}")
    dst = LOCAL_DATA / "all_runs_features_model_ready.parquet"
    if dst.exists():
        log(f"using cached merged parquet ({len(chunks)} chunks → {dst})")
        return dst
    log(f"merging {len(chunks)} chunks from {src_dir} …")

    # Pass 1: scan schemas only (fast metadata reads, no data) to find widest type per column.
    # The streaming funnel uses aggressive per-chunk downcast, so the same column can be
    # int64 in one chunk and float64 in another depending on the values present.
    log("  scanning chunk schemas …")
    col_types: dict = {}
    col_order: list = []
    for chunk in chunks:
        for field in pq.read_schema(chunk):
            t = field.type
            if field.name not in col_types:
                col_types[field.name] = t
                col_order.append(field.name)
            else:
                existing = col_types[field.name]
                # Any float beats any int; wider beats narrower within the same kind
                if pa.types.is_floating(existing) or pa.types.is_floating(t):
                    col_types[field.name] = pa.float64()
                elif pa.types.is_integer(existing) and pa.types.is_integer(t):
                    col_types[field.name] = pa.int64()
    unified = pa.schema([pa.field(n, col_types[n]) for n in col_order])
    log(f"  unified schema: {len(unified)} columns")

    # Pass 2: read each chunk, cast to unified schema, write incrementally.
    writer = None
    total_rows = 0
    try:
        for i, chunk in enumerate(chunks):
            df = pd.read_parquet(chunk)
            table = pa.Table.from_pandas(df, preserve_index=False)
            del df
            # Cast every column to the unified type
            casted = {}
            for field in unified:
                if field.name in table.schema.names:
                    col = table.column(field.name)
                    if col.type != field.type:
                        col = col.cast(field.type, safe=False)
                    casted[field.name] = col
                else:
                    casted[field.name] = pa.nulls(len(table), type=field.type)
            table = pa.table(casted, schema=unified)
            if writer is None:
                writer = pq.ParquetWriter(str(dst), unified)
            writer.write_table(table)
            total_rows += len(table)
            del table
            if (i + 1) % 20 == 0:
                log(f"  merged {i+1}/{len(chunks)} chunks ({total_rows:,} rows) …")
    finally:
        if writer:
            writer.close()
    log(f"merged {total_rows:,} rows → {dst}")
    return dst

# Streaming chunks (5,080 runs) take priority over old single-file export (3,140 runs).
CHUNKS_DIR = CLEAN_ROOT / "all_runs_features_model_ready_chunks"
if CHUNKS_DIR.exists() and any(CHUNKS_DIR.glob("part_*.parquet")):
    log(f"streaming chunks found: {CHUNKS_DIR}")
    FEATURE_PATH = stage_chunks(CHUNKS_DIR)
else:
    feature_candidates = []
    if LATEST_EXPORT:
        feature_candidates += [
            LATEST_EXPORT / "model_ready" / "all_runs_features_after_stall_cut_df.parquet",
            LATEST_EXPORT / "engineered"  / "all_runs_features_df.parquet",
        ]
    feature_candidates += [
        CLEAN_ROOT / "latest_export" / "model_ready" / "all_runs_features_after_stall_cut_df.parquet",
        CLEAN_ROOT / "exports" / "latest_export" / "model_ready" / "all_runs_features_after_stall_cut_df.parquet",
        CLEAN_ROOT / "all_runs_features_after_stall_cut.parquet",
        CLEAN_ROOT / "all_runs_features_after_stall_cut_df.parquet",
        CLEAN_ROOT / "all_runs_features_df.parquet",
    ]
    DRIVE_FEAT   = first_existing(feature_candidates, "features")
    FEATURE_PATH = stage(DRIVE_FEAT)

DRIVE_LABEL = first_existing(label_candidates, "labels")
LABEL_PATH  = stage(DRIVE_LABEL)

log(f"features: {FEATURE_PATH}")
log(f"labels:   {LABEL_PATH}")

# =============================================================================
# Load and merge features + labels
# =============================================================================
log("loading feature table …")
# Tight column filter: only load what the model actually uses.
# 2D pre-allocation: pandas wraps a single numpy array as one block — no consolidation copy.
# Together these keep peak RAM under ~3 GB (vs ~9 GB with naive load + pd.concat).
import pyarrow as _pa
import pyarrow.parquet as _pq
import numpy as _np
# Mirrors NODE_PATTERNS exactly so single-underscore cols (e.g. effusion_temp_K) aren't dropped.
_NODE_COL_PREFIXES = (
    "orbit__", "orbit_", "t_orbit", "theta_", "altitude", "sunlight", "eclipse",
    "solar_array__", "solararray__", "solar_",
    "array_gimbal__", "arraygimbal__", "array_gimbal_", "arraygimbal_",
    "battery__", "battery_",
    "battery_thermal__", "batterythermal__", "battery_thermal_", "batterythermal_",
    "power_bus__", "powerbus__", "power_bus_", "total_requested_power", "total_delivered_power",
    "heater_bank__", "heaterbank__", "heater_bank_",
    "effusion_cell__", "effusioncell__", "effusion_",
    "source_inventory__", "sourceinventory__", "source_inventory_", "sourceinventory_",
    "substrate__", "substrate_",
    "cryo_panel__", "cryopanel__", "cryo_panel_", "cryopanel_",
    "radiator__", "radiator_",
    "simulation_engine__", "simulationengine__", "simulation_",
)
_CONTEXT_PREFIXES = ("process_state__", "processstate__", "schedule_state__", "schedulestate__")
# Metadata cols needed for split / windowing / label join — everything else discarded.
_META_COLS = {
    "run_id", "run_id_label", "config_key", "config_label", "config_id", "config_name",
    "config_label_from_funnel", "canonical_tick", "tick", "simulation_tick",
}
_pq_schema = _pq.read_schema(FEATURE_PATH)
def _want(c):
    cl = c.lower()
    return (c in _META_COLS
            or any(cl.startswith(p) for p in _NODE_COL_PREFIXES)
            or any(cl.startswith(p) for p in _CONTEXT_PREFIXES))
_keep_cols = [c for c in _pq_schema.names if _want(c)]
log(f"  selecting {len(_keep_cols)}/{len(_pq_schema.names)} columns …")

_pf = _pq.ParquetFile(FEATURE_PATH)
_nrows = _pf.metadata.num_rows

# Determine which cols are string vs numeric from the schema (no data read needed).
_str_cols = [
    c for c in _keep_cols
    if _pa.types.is_string(_pq_schema.field(c).type)
    or _pa.types.is_large_string(_pq_schema.field(c).type)
]
_num_cols = [c for c in _keep_cols if c not in _str_cols]

# Pre-allocate: ONE 2D float32 array for all numeric cols (no per-column dict → no
# pandas consolidation copy later). String cols kept as object arrays separately.
_num_arr2d = _np.empty((_nrows, len(_num_cols)), dtype=_np.float32)
_str_arrs  = {c: _np.empty(_nrows, dtype=object) for c in _str_cols}
_num_idx   = {c: j for j, c in enumerate(_num_cols)}

_pf2 = _pq.ParquetFile(FEATURE_PATH)
_row = 0
for _batch in _pf2.iter_batches(batch_size=250_000, columns=_keep_cols):
    _chunk = _batch.to_pandas()
    _n = len(_chunk)
    for c in _num_cols:
        _num_arr2d[_row:_row + _n, _num_idx[c]] = _chunk[c].values
    for c in _str_cols:
        _str_arrs[c][_row:_row + _n] = _chunk[c].values
    del _chunk
    _row += _n
    log(f"  loaded {_row:,} rows …")
gc.collect()

# Build DataFrame: 2D array → one pandas block, zero copy; string cols added individually.
features_df = pd.DataFrame(_num_arr2d, columns=_num_cols)
del _num_arr2d
for _sc in _str_cols:
    features_df[_sc] = _str_arrs[_sc]
del _str_arrs, _pf, _pf2, _pq_schema
gc.collect()

run_funnel_df = pd.read_csv(LABEL_PATH)
log(f"features shape: {features_df.shape}")
log(f"labels shape:   {run_funnel_df.shape}")

NO_FAILURE_REASONS = {
    "", "nan", "none", "null", "no_failure", "no failure",
    "nofailure", "success", "finished", "complete", "completed",
    "ok", "passed", "pass",
}

def pick_col(df, candidates, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    if required:
        raise ValueError(f"Could not find any of: {candidates}")
    return None

def clean_key(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s

def parse_bool_like(series):
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[s.isin({"1","true","t","yes","y","failed","failure","fail"})] = 1
    out.loc[s.isin({"0","false","f","no","n","success","ok","passed","pass","no_failure"})] = 0
    nums = pd.to_numeric(s, errors="coerce")
    out.loc[nums.notna()] = (nums.loc[nums.notna()] != 0).astype(int)
    return out

RUN_ID_COL = pick_col(features_df, ["run_id"])
CONFIG_COL  = pick_col(features_df, ["config_label","config_id","config","config_name"])
TICK_COL    = pick_col(features_df, ["canonical_tick","tick","simulation_tick"])

# Drop string columns that aren't needed as features or identifiers.
# Each string column costs ~100 MB at 2.2M rows (pandas object dtype).
_keep_str = {RUN_ID_COL, CONFIG_COL, "run_id", "config_id", "config_label", "config_name"}
_drop_str = [c for c in features_df.select_dtypes("object").columns if c not in _keep_str]
if _drop_str:
    features_df.drop(columns=_drop_str, inplace=True)
    gc.collect()
    log(f"dropped {len(_drop_str)} string columns to reduce RAM")

# pd.concat already produced a fresh copy — no need to copy again.
features_df["canonical_tick"] = pd.to_numeric(features_df[TICK_COL], errors="coerce")
features_df[RUN_ID_COL]  = features_df[RUN_ID_COL].astype(str)
features_df[CONFIG_COL]  = features_df[CONFIG_COL].astype(str)
features_df["_key"]      = features_df[RUN_ID_COL].map(clean_key)

label_like_cols = [
    "funnel_failed","funnel_failure_reason","funnel_failure_tick",
    "funnel_failure_step","funnel_failure_group","funnel_failure_label",
    "coarse_cause","y_fail_run","cause_idx","run_id_label",
]
features_df.drop(columns=[c for c in label_like_cols if c in features_df.columns], inplace=True)

label_run_id_col  = pick_col(run_funnel_df, ["run_id"])
label_config_col  = pick_col(run_funnel_df, ["config_label","config_id","config","config_name"], required=False)
label_failed_col  = pick_col(run_funnel_df, ["funnel_failed","run_failed","failed","has_failure","is_failure","y_fail_run","y_fail"], required=False)
label_reason_col  = pick_col(run_funnel_df, ["funnel_failure_reason","failure_reason","final_failure_reason","reason","failure_type","coarse_cause"], required=False)
label_tick_col    = pick_col(run_funnel_df, ["funnel_failure_tick","failure_tick","first_failure_tick","cutoff_tick","fail_tick"], required=False)

rl = run_funnel_df.copy()
rl[label_run_id_col] = rl[label_run_id_col].astype(str)
rl["_key"] = rl[label_run_id_col].map(clean_key)
rl["lbl_run_id"] = rl[label_run_id_col].astype(str)
if label_config_col:
    rl["lbl_config_key"] = rl[label_config_col].astype(str)
rl["lbl_reason"] = rl[label_reason_col].astype(str).str.strip() if label_reason_col else "no_failure"
rl["lbl_tick"]   = pd.to_numeric(rl[label_tick_col], errors="coerce") if label_tick_col else np.nan

reason_norm = rl["lbl_reason"].astype(str).str.strip().str.lower()
reason_failed = (~reason_norm.isin(NO_FAILURE_REASONS)).astype(int)
tick_failed   = rl["lbl_tick"].notna().astype(int)

if label_failed_col:
    parsed = parse_bool_like(rl[label_failed_col])
    rl["lbl_failed"] = parsed.fillna(reason_failed).fillna(tick_failed).astype(int)
else:
    rl["lbl_failed"] = np.maximum(reason_failed, tick_failed).astype(int)

rl.loc[reason_norm.isin(NO_FAILURE_REASONS) & rl["lbl_tick"].isna(), "lbl_failed"] = 0
rl.loc[rl["lbl_failed"].eq(0), "lbl_reason"] = "no_failure"

keep = ["_key","lbl_run_id","lbl_failed","lbl_reason","lbl_tick"]
if "lbl_config_key" in rl.columns:
    keep.append("lbl_config_key")
rl = rl[keep].drop_duplicates("_key")

# Map label columns in-place instead of merge to avoid doubling 3 GB in RAM.
_rl_idx = rl.set_index("_key")
for _lbl_col in [c for c in rl.columns if c != "_key"]:
    features_df[_lbl_col] = features_df["_key"].map(_rl_idx[_lbl_col])
del _rl_idx, rl
gc.collect()
df = features_df  # same object, no copy
df["run_id"]     = df[RUN_ID_COL].astype(str)
df["config_key"] = df[CONFIG_COL].astype(str)
df["funnel_failed"]        = pd.to_numeric(df["lbl_failed"], errors="coerce").fillna(0).astype(int)
df["funnel_failure_tick"]  = pd.to_numeric(df["lbl_tick"],   errors="coerce")
df["funnel_failure_reason"] = df["lbl_reason"].fillna("no_failure").astype(str)

reason_norm_df = df["funnel_failure_reason"].astype(str).str.strip().str.lower()
df.loc[reason_norm_df.isin(NO_FAILURE_REASONS) & df["funnel_failure_tick"].isna(), "funnel_failed"] = 0
df.loc[df["funnel_failed"].eq(0), "funnel_failure_reason"] = "no_failure"

log(f"merged df shape: {df.shape}  unique runs: {df['run_id'].nunique()}  unique configs: {df['config_key'].nunique()}")

# =============================================================================
# Coarse labels + run-level meta
# =============================================================================
COARSE_CAUSE_MAP = {
    "no_failure":                          "no_failure",
    "stall_60plus_cutoff":                 "stall",
    "effusion_underflux_cap_hit":          "effusion_underflux",
    "effusion_underflux_cap_one_same_tick":"effusion_underflux",
    "effusion_undertemp_cap_hit":          "effusion_undertemp",
    "effusion_undertemp_cap_one_same_tick":"effusion_undertemp",
    "substrate_undertemp_cap_hit":         "substrate_undertemp",
    "battery_soc_below_5pct":              "battery_power",
    "logged_failure_unclassified":         "unknown_failure",
}

def map_coarse_cause(reason, failed):
    reason = str(reason).strip() if pd.notna(reason) else "no_failure"
    if int(failed) == 0 or reason.lower() in NO_FAILURE_REASONS:
        return "no_failure"
    return COARSE_CAUSE_MAP.get(reason, "unknown_failure")

run_meta = (
    df.groupby("run_id", as_index=False)
      .agg(
          config_key=("config_key","first"),
          funnel_failed=("funnel_failed","max"),
          funnel_failure_reason=("funnel_failure_reason","first"),
          funnel_failure_tick=("funnel_failure_tick","min"),
          min_tick=("canonical_tick","min"),
          max_tick=("canonical_tick","max"),
          n_ticks=("canonical_tick","count"),
      )
)
run_meta["coarse_cause"] = [map_coarse_cause(r, f) for r, f in zip(run_meta["funnel_failure_reason"], run_meta["funnel_failed"])]
run_meta["y_fail_run"]   = (run_meta["coarse_cause"] != "no_failure").astype(int)

bad_tick = run_meta["y_fail_run"].eq(1) & run_meta["funnel_failure_tick"].isna()
if bad_tick.any():
    log(f"dropping {int(bad_tick.sum())} positive runs with missing failure tick")
    valid = set(run_meta.loc[~bad_tick, "run_id"])
    df       = df[df["run_id"].isin(valid)].copy()
    run_meta = run_meta[~bad_tick].reset_index(drop=True)

cause_names = sorted(run_meta["coarse_cause"].unique().tolist())
cause_names = ["no_failure"] + [c for c in cause_names if c != "no_failure"]
cause_to_idx = {c: i for i, c in enumerate(cause_names)}
idx_to_cause = {i: c for c, i in cause_to_idx.items()}
run_meta["cause_idx"] = run_meta["coarse_cause"].map(cause_to_idx).astype(int)

# v10 hierarchical cause labels. The flat cause head is kept, but we also
# supervise a broad family head and a small effusion subtype head.
CAUSE_TTF_HEAD_NAMES = ["battery_power", "effusion_underflux", "effusion_undertemp", "stall", "substrate_undertemp"]
cause_ttf_to_idx = {c: i for i, c in enumerate(CAUSE_TTF_HEAD_NAMES)}
idx_to_cause_ttf = {i: c for c, i in cause_ttf_to_idx.items()}

def cause_to_family_name(cause_name):
    if cause_name == "no_failure":
        return "no_failure"
    if str(cause_name).startswith("battery"):
        return "battery"
    if str(cause_name).startswith("effusion"):
        return "effusion"
    if str(cause_name).startswith("substrate"):
        return "substrate"
    if str(cause_name).startswith("stall"):
        return "stall"
    return "unknown_failure"

family_names = ["no_failure", "battery", "effusion", "substrate", "stall", "unknown_failure"]
family_to_idx = {c: i for i, c in enumerate(family_names)}
idx_to_family = {i: c for c, i in family_to_idx.items()}

effusion_subtype_names = ["not_effusion", "effusion_underflux", "effusion_undertemp"]
effusion_subtype_to_idx = {c: i for i, c in enumerate(effusion_subtype_names)}
idx_to_effusion_subtype = {i: c for c, i in effusion_subtype_to_idx.items()}

run_meta["family_name"] = run_meta["coarse_cause"].map(cause_to_family_name)
run_meta["family_idx"] = run_meta["family_name"].map(family_to_idx).fillna(family_to_idx["unknown_failure"]).astype(int)
run_meta["effusion_subtype_idx"] = run_meta["coarse_cause"].map(lambda c: effusion_subtype_to_idx.get(c, 0)).astype(int)
run_meta["cause_ttf_head_idx"] = run_meta["coarse_cause"].map(lambda c: cause_ttf_to_idx.get(c, -1)).astype(int)

total_runs  = len(run_meta)
failed_runs = int(run_meta["y_fail_run"].sum())
alive_runs  = total_runs - failed_runs
log(f"runs: total={total_runs}  failed={failed_runs}  alive={alive_runs}  rate={failed_runs/total_runs:.3f}")
log(f"cause distribution:\n{run_meta['coarse_cause'].value_counts().to_string()}")


# =============================================================================
# Safe derivative/rate features
# =============================================================================
# These features use only current and past ticks within each run. They are added
# before feature selection so Model A and Model B can use them like normal node
# features. No failure flags, streak counters, abort flags, final labels, or
# future values are used.

def _find_first_matching_col(columns, required_terms=(), any_terms=(), banned_terms=()):
    matches = []
    for col in columns:
        nc = norm_col(col) if "norm_col" in globals() else str(col).lower()
        if any(bt in nc for bt in banned_terms):
            continue
        if all(rt in nc for rt in required_terms) and (not any_terms or any(at in nc for at in any_terms)):
            matches.append(col)
    return sorted(matches, key=lambda c: len(str(c)))[0] if matches else None


def _find_node_col(columns, node_aliases, any_terms, banned_terms=()):
    """Find a telemetry field across snake_case and compact node prefixes."""
    for alias in node_aliases:
        found = _find_first_matching_col(
            columns, required_terms=(alias,), any_terms=any_terms,
            banned_terms=banned_terms,
        )
        if found:
            return found
    return None


def _past_slope(series, periods):
    s = pd.to_numeric(series, errors="coerce").astype("float32")
    return (s - s.shift(periods)) / float(periods)


def _past_rolling_sum(series, window):
    s = pd.to_numeric(series, errors="coerce").astype("float32")
    return s.rolling(window=window, min_periods=1).sum().astype("float32")


def add_safe_engineered_features(df_in):
    log("adding v11 past-only derivative/rate features ...")
    df_out = df_in.sort_values(["run_id", "canonical_tick"]).copy()
    cols = list(df_out.columns)

    battery_soc_col = (
        _find_first_matching_col(cols, required_terms=("battery",), any_terms=("soc", "state_of_charge"), banned_terms=("thermal",))
        or _find_first_matching_col(cols, required_terms=("battery",), any_terms=("charge_wh", "charge"), banned_terms=("thermal",))
    )
    req_col = _find_first_matching_col(cols, any_terms=("total_requested_power", "requested_power", "request_w"))
    deliv_col = _find_first_matching_col(cols, any_terms=("total_delivered_power", "delivered_power", "delivered_w"))

    eff_temp_col = _find_first_matching_col(cols, required_terms=("effusion",), any_terms=("temp_k", "temperature_k", "temp"))
    eff_target_col = _find_first_matching_col(cols, required_terms=("effusion", "target"), any_terms=("k", "temp"))
    flux_col = _find_first_matching_col(cols, required_terms=("flux",), banned_terms=("fail", "failure", "streak", "label"))

    sub_temp_col = _find_first_matching_col(cols, required_terms=("substrate",), any_terms=("temp_k", "temperature_k", "temp"))
    sub_target_col = _find_first_matching_col(cols, required_terms=("substrate", "target"), any_terms=("k", "temp"))

    batt_t_col = _find_node_col(cols, ("battery_thermal", "batterythermal"), ("t_batt_k", "pack_temp", "batt_temp"))
    batt_sink_col = _find_node_col(cols, ("battery_thermal", "batterythermal"), ("t_sink_k", "sink_temp"))
    radiator_t_col = _find_node_col(cols, ("radiator",), ("t_loop_k", "loop_temp"))
    radiator_load_col = _find_node_col(cols, ("radiator",), ("q_load_w", "heat_load"))
    radiator_reject_col = _find_node_col(cols, ("radiator",), ("q_reject_w", "heat_reject"))
    source_frac_col = _find_node_col(cols, ("source_inventory", "sourceinventory"), ("remaining_frac",))
    gimbal_eff_col = _find_node_col(cols, ("array_gimbal", "arraygimbal"), ("pointing_eff",))
    gimbal_err_col = _find_node_col(cols, ("array_gimbal", "arraygimbal"), ("pointing_err",))
    cryo_mass_col = _find_node_col(cols, ("cryo_panel", "cryopanel"), ("adsorbed_g",))
    cryo_req_col = _find_node_col(cols, ("cryo_panel", "cryopanel"), ("power_req_w",))
    cryo_grant_col = _find_node_col(cols, ("cryo_panel", "cryopanel"), ("power_granted_w",))

    grp = df_out.groupby("run_id", sort=False)

    if battery_soc_col:
        df_out["battery_soc_slope_5"] = grp[battery_soc_col].transform(lambda s: _past_slope(s, 5))
        df_out["battery_soc_slope_20"] = grp[battery_soc_col].transform(lambda s: _past_slope(s, 20))
        log(f"  battery slope source: {battery_soc_col}")
    else:
        log("  battery SOC/charge source not found; battery slopes skipped")

    if req_col and deliv_col:
        df_out["power_deficit_w"] = pd.to_numeric(df_out[req_col], errors="coerce") - pd.to_numeric(df_out[deliv_col], errors="coerce")
        df_out["power_deficit_rolling_sum_10"] = grp["power_deficit_w"].transform(lambda s: _past_rolling_sum(s, 10))
        df_out["power_deficit_rolling_sum_30"] = grp["power_deficit_w"].transform(lambda s: _past_rolling_sum(s, 30))
        log(f"  power deficit sources: requested={req_col}, delivered={deliv_col}")
    else:
        log("  requested/delivered power sources not found; power deficit features skipped")

    if eff_temp_col and eff_target_col:
        df_out["effusion_temp_error"] = pd.to_numeric(df_out[eff_temp_col], errors="coerce") - pd.to_numeric(df_out[eff_target_col], errors="coerce")
        df_out["effusion_temp_error_slope_5"] = grp["effusion_temp_error"].transform(lambda s: _past_slope(s, 5))
        df_out["effusion_temp_error_slope_20"] = grp["effusion_temp_error"].transform(lambda s: _past_slope(s, 20))
        log(f"  effusion temp error sources: temp={eff_temp_col}, target={eff_target_col}")
    else:
        log("  effusion temp/target sources not found; effusion temp error skipped")

    if flux_col:
        flux = pd.to_numeric(df_out[flux_col], errors="coerce").astype("float32")
        # Use a past rolling median as a local safe normalizer, not a future/global label.
        denom = grp[flux_col].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(60, min_periods=5).median()).replace(0, np.nan)
        df_out["flux_ratio"] = (flux / denom).astype("float32")
        df_out["flux_ratio_slope_5"] = grp["flux_ratio"].transform(lambda s: _past_slope(s, 5))
        df_out["flux_ratio_slope_20"] = grp["flux_ratio"].transform(lambda s: _past_slope(s, 20))
        log(f"  flux ratio source: {flux_col}")
    else:
        log("  flux source not found; flux ratio features skipped")

    if sub_temp_col and sub_target_col:
        df_out["substrate_temp_error"] = pd.to_numeric(df_out[sub_temp_col], errors="coerce") - pd.to_numeric(df_out[sub_target_col], errors="coerce")
        df_out["substrate_temp_error_slope_5"] = grp["substrate_temp_error"].transform(lambda s: _past_slope(s, 5))
        df_out["substrate_temp_error_slope_20"] = grp["substrate_temp_error"].transform(lambda s: _past_slope(s, 20))
        log(f"  substrate temp error sources: temp={sub_temp_col}, target={sub_target_col}")
    else:
        log("  substrate temp/target sources not found; substrate temp error skipped")

    # Expansion-node trends expose slow degradation and coupling margins that a
    # 64-tick temporal window otherwise has to infer from noisier raw channels.
    for out_prefix, source_col in [
        ("battery_thermal_pack_temp", batt_t_col),
        ("radiator_loop_temp", radiator_t_col),
        ("source_inventory_remaining_frac", source_frac_col),
        ("array_gimbal_pointing_eff", gimbal_eff_col),
        ("array_gimbal_pointing_err", gimbal_err_col),
        ("cryo_panel_adsorbed_mass", cryo_mass_col),
    ]:
        if source_col:
            df_out[f"{out_prefix}_slope_5"] = grp[source_col].transform(lambda s: _past_slope(s, 5))
            df_out[f"{out_prefix}_slope_20"] = grp[source_col].transform(lambda s: _past_slope(s, 20))
            log(f"  {out_prefix} slopes source: {source_col}")

    if batt_t_col and batt_sink_col:
        df_out["battery_thermal_sink_delta_k"] = (
            pd.to_numeric(df_out[batt_t_col], errors="coerce")
            - pd.to_numeric(df_out[batt_sink_col], errors="coerce")
        )
    if radiator_load_col and radiator_reject_col:
        df_out["radiator_heat_balance_w"] = (
            pd.to_numeric(df_out[radiator_load_col], errors="coerce")
            - pd.to_numeric(df_out[radiator_reject_col], errors="coerce")
        )
        df_out["radiator_heat_balance_rolling_sum_20"] = grp["radiator_heat_balance_w"].transform(
            lambda s: _past_rolling_sum(s, 20)
        )
    if cryo_req_col and cryo_grant_col:
        df_out["cryo_panel_power_deficit_w"] = (
            pd.to_numeric(df_out[cryo_req_col], errors="coerce")
            - pd.to_numeric(df_out[cryo_grant_col], errors="coerce")
        )
        df_out["cryo_panel_power_deficit_rolling_sum_20"] = grp["cryo_panel_power_deficit_w"].transform(
            lambda s: _past_rolling_sum(s, 20)
        )

    new_cols = [
        "battery_soc_slope_5", "battery_soc_slope_20",
        "power_deficit_w", "power_deficit_rolling_sum_10", "power_deficit_rolling_sum_30",
        "effusion_temp_error", "effusion_temp_error_slope_5", "effusion_temp_error_slope_20",
        "flux_ratio", "flux_ratio_slope_5", "flux_ratio_slope_20",
        "substrate_temp_error", "substrate_temp_error_slope_5", "substrate_temp_error_slope_20",
        "battery_thermal_pack_temp_slope_5", "battery_thermal_pack_temp_slope_20",
        "battery_thermal_sink_delta_k",
        "radiator_loop_temp_slope_5", "radiator_loop_temp_slope_20",
        "radiator_heat_balance_w", "radiator_heat_balance_rolling_sum_20",
        "source_inventory_remaining_frac_slope_5", "source_inventory_remaining_frac_slope_20",
        "array_gimbal_pointing_eff_slope_5", "array_gimbal_pointing_eff_slope_20",
        "array_gimbal_pointing_err_slope_5", "array_gimbal_pointing_err_slope_20",
        "cryo_panel_adsorbed_mass_slope_5", "cryo_panel_adsorbed_mass_slope_20",
        "cryo_panel_power_deficit_w", "cryo_panel_power_deficit_rolling_sum_20",
    ]
    for c in new_cols:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    added = [c for c in new_cols if c in df_out.columns]
    log(f"  engineered features added: {added}")
    return df_out.reset_index(drop=True)

df = add_safe_engineered_features(df)
gc.collect()


# =============================================================================
# Feature selection
# =============================================================================
import re as _re

LEGACY_NODES = [
    "orbit", "solar_array", "battery", "power_bus", "heater_bank",
    "effusion_cell", "substrate", "simulation_engine",
]
EXPANSION_NODES = [
    "array_gimbal", "battery_thermal", "cryo_panel", "radiator",
    "source_inventory",
]
PHYSICAL_NODES = [
    "orbit", "solar_array", "array_gimbal", "battery", "battery_thermal",
    "power_bus", "heater_bank", "effusion_cell", "source_inventory",
    "substrate", "cryo_panel", "radiator", "simulation_engine",
]

NODE_PATTERNS = {
    "orbit":             [r"^orbit__", r"^orbit_", r"^t_orbit", r"^theta_", r"^altitude", r"^sunlight", r"^eclipse"],
    "solar_array":       [r"^solar_array__", r"^solararray__", r"^solar_"],
    "array_gimbal":      [r"^array_gimbal__", r"^arraygimbal__", r"^array_gimbal_", r"^arraygimbal_"],
    "battery":           [r"^battery__", r"^battery_"],
    "battery_thermal":   [r"^battery_thermal__", r"^batterythermal__", r"^battery_thermal_", r"^batterythermal_"],
    "power_bus":         [r"^power_bus__", r"^powerbus__", r"^power_bus_", r"^total_requested_power", r"^total_delivered_power"],
    "heater_bank":       [r"^heater_bank__", r"^heaterbank__", r"^heater_bank_"],
    "effusion_cell":     [r"^effusion_cell__", r"^effusioncell__", r"^effusion_", r"^flux_ratio"],
    "source_inventory":  [r"^source_inventory__", r"^sourceinventory__", r"^source_inventory_", r"^sourceinventory_"],
    "substrate":         [r"^substrate__", r"^substrate_"],
    "cryo_panel":        [r"^cryo_panel__", r"^cryopanel__", r"^cryo_panel_", r"^cryopanel_"],
    "radiator":          [r"^radiator__", r"^radiator_"],
    "simulation_engine": [r"^simulation_engine__", r"^simulationengine__", r"^simulation_"],
}

PROCESS_SAFE_BASE = [
    "effusionDemand_W","substrateDemand_W","effusion_temp_K","effusion_target_K",
    "effusion_T_env_eff_K","effusion_solar_scale","effusion_P_solar_abs_W",
    "substrate_temp_K","substrate_target_K","substrate_T_env_eff_K",
    "substrate_solar_scale","substrate_P_solar_abs_W",
    "effusion_delivered_W","substrate_delivered_W","raw_job_flux_cm2s",
]
SCHEDULE_SAFE_BASE = [
    "phase_ticks_completed","live_ticks_completed","delay_from_requested_start",
    "queued_active","warmup_active","cooldown_active","thermal_prep_active",
]
LEAKAGE_PATTERNS = [
    r"^funnel_", r"^failed_", r"^stall_", r"_failure_", r"_failed",
    r"aborted_active", r"done_active", r"job_failed",
    r"(^|_)tick$", r"^tick$", r"time_s$", r"^time_s$",
    r"actual_.*_end_tick", r"underflux_streak", r"temp_miss_streak",
    r"failure_reason", r"coarse_cause", r"cause_idx", r"y_fail", r"y_ttf",
]
ID_AND_LABEL_COLS = {
    "run_id","run_id_label","config_key","config_label_from_funnel",
    "canonical_tick","funnel_failed","funnel_failure_reason","funnel_failure_tick",
}

def norm_col(c): return _re.sub(r"[^a-zA-Z0-9]+","_",str(c)).strip("_").lower()
def is_numeric(df, col): return col not in ID_AND_LABEL_COLS and (pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col],errors="coerce").notna().any())
def is_leakage(col): return any(_re.search(p, norm_col(col)) for p in LEAKAGE_PATTERNS)

def find_safe_context_columns(all_cols):
    process_bases  = {norm_col(x) for x in PROCESS_SAFE_BASE}
    schedule_bases = {norm_col(x) for x in SCHEDULE_SAFE_BASE}
    proc, sched = [], []
    for col in all_cols:
        nc = norm_col(col)
        tails = {nc, "_".join(nc.split("_")[-1:]), "_".join(nc.split("_")[-2:]), "_".join(nc.split("_")[-3:]), "_".join(nc.split("_")[-4:])}
        if (nc.startswith("process_state_") or nc.startswith("processstate_")) and tails & process_bases:
            proc.append(col)
        if (nc.startswith("schedule_state_") or nc.startswith("schedulestate_")) and tails & schedule_bases:
            sched.append(col)
    return sorted(proc), sorted(sched)

def select_node_columns(df):
    node_cols = {n: [] for n in PHYSICAL_NODES}
    for col in df.columns:
        if col in ID_AND_LABEL_COLS or is_leakage(col) or not is_numeric(df, col):
            continue
        nc = norm_col(col)
        if nc.startswith("process_state_") or nc.startswith("processstate_"):
            continue
        if nc.startswith("schedule_state_") or nc.startswith("schedulestate_"):
            continue
        matched = [n for n, pats in NODE_PATTERNS.items() if any(_re.search(p, nc) for p in pats)]
        if len(matched) == 1:
            node_cols[matched[0]].append(col)
        elif len(matched) > 1:
            node_cols[max(matched, key=len)].append(col)
    for n in node_cols:
        node_cols[n] = sorted(set(node_cols[n]))
    return node_cols

node_cols_by_node = select_node_columns(df)
proc_cols, sched_cols = find_safe_context_columns(list(df.columns))
context_cols_b = sorted({c for c in proc_cols + sched_cols if c not in ID_AND_LABEL_COLS and not is_leakage(c) and is_numeric(df, c)})

log("node feature counts:")
for n in PHYSICAL_NODES:
    log(f"  {n:20s} {len(node_cols_by_node[n])}")
log(f"safe context cols (Model B): {len(context_cols_b)}")
missing_expansion_nodes = [n for n in EXPANSION_NODES if not node_cols_by_node[n]]
if missing_expansion_nodes:
    log(
        "WARNING: expansion nodes with no selected telemetry: "
        + ", ".join(missing_expansion_nodes)
        + ". v11 remains backward compatible, but retraining on the new export is required to use them."
    )

# =============================================================================
# Config-level split
# =============================================================================
def split_configs(run_meta, seed):
    # Dominant cause per config — used for stratification so every cause type
    # (including rare ones like substrate_undertemp) appears in all three splits.
    cfg_cause = (
        run_meta[run_meta["coarse_cause"] != "no_failure"]
        .groupby("config_key")["coarse_cause"]
        .agg(lambda x: x.value_counts().index[0])  # most common failure cause
        .reset_index()
        .rename(columns={"coarse_cause": "dominant_cause"})
    )
    cfg_lvl = run_meta.groupby("config_key", as_index=False).agg(
        any_failure=("y_fail_run", "max"), n_runs=("run_id", "nunique"))
    cfg_lvl = cfg_lvl.merge(cfg_cause, on="config_key", how="left")
    cfg_lvl["dominant_cause"] = cfg_lvl["dominant_cause"].fillna("no_failure")

    # Stratify by dominant cause; collapse rare causes to "other" so sklearn doesn't error
    cause_counts = cfg_lvl["dominant_cause"].value_counts()
    cfg_lvl["strat_label"] = cfg_lvl["dominant_cause"].apply(
        lambda c: c if cause_counts[c] >= 6 else "other")

    configs = cfg_lvl["config_key"].tolist()
    strat   = cfg_lvl["strat_label"].tolist()
    strat   = strat if len(set(strat)) > 1 and min(pd.Series(strat).value_counts()) >= 2 else None

    train_c, temp_c = train_test_split(configs, test_size=0.30, random_state=seed, stratify=strat)

    temp_df   = cfg_lvl[cfg_lvl["config_key"].isin(temp_c)]
    strat2    = temp_df["strat_label"].tolist()
    strat2    = strat2 if len(set(strat2)) > 1 and min(pd.Series(strat2).value_counts()) >= 2 else None
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=seed, stratify=strat2)

    train_c_set = set(train_c)
    val_c_set   = set(val_c)
    test_c_set  = set(test_c)

    # Guarantee every failure cause has ≥1 config in test and ≥1 in val.
    # If a cause is missing from test/val, steal one config from train.
    rng = np.random.default_rng(seed)
    failure_causes = [c for c in cause_counts.index if c != "no_failure"]
    for cause in failure_causes:
        cause_cfg = cfg_lvl[cfg_lvl["dominant_cause"] == cause]["config_key"].tolist()
        for target_set, other_sets in [
            (test_c_set, [train_c_set, val_c_set]),
            (val_c_set,  [train_c_set]),
        ]:
            if not any(c in target_set for c in cause_cfg):
                # Find a candidate in train (preferred) or the other set
                candidates = [c for c in cause_cfg if any(c in s for s in other_sets)]
                if candidates:
                    chosen = rng.choice(candidates)
                    # Remove from whichever set holds it
                    for s in other_sets:
                        s.discard(chosen)
                    target_set.add(chosen)
                    log(f"  [split] forced {cause} config {chosen!r} → "
                        f"{'test' if target_set is test_c_set else 'val'}")

    log("config split by dominant cause:")
    for split_name_, split_set in [("train", train_c_set), ("val", val_c_set), ("test", test_c_set)]:
        sc = cfg_lvl[cfg_lvl["config_key"].isin(split_set)]["dominant_cause"].value_counts().to_dict()
        log(f"  {split_name_:6s}: {sc}")

    return train_c_set, val_c_set, test_c_set, cfg_lvl

TRAIN_CONFIGS, VAL_CONFIGS, TEST_CONFIGS, config_level_df = split_configs(run_meta, SEED)

def split_name(cfg):
    if cfg in TRAIN_CONFIGS: return "train"
    if cfg in VAL_CONFIGS:   return "val"
    if cfg in TEST_CONFIGS:  return "test"
    return "unknown"

run_meta["split"] = run_meta["config_key"].map(split_name)
split_run_sets = {s: set(run_meta.loc[run_meta["split"].eq(s), "run_id"]) for s in ["train","val","test"]}

log("config split:")
for s in ["train","val","test"]:
    cfgs = run_meta[run_meta["split"].eq(s)]["config_key"].nunique()
    runs = len(split_run_sets[s])
    fails = int(run_meta[run_meta["split"].eq(s)]["y_fail_run"].sum())
    log(f"  {s:6s}  configs={cfgs}  runs={runs}  failures={fails}")

# =============================================================================
# Window index
# =============================================================================
WINDOW_LEN   = CFG["window_len"]
WINDOW_STRIDE = CFG["window_stride"]
NEAR_FAILURE_STRIDE = int(CFG.get("near_failure_stride", 1))
NEAR_FAILURE_STRIDE_WINDOW_TICKS = float(CFG.get("near_failure_stride_window_ticks", 120.0))
TTF_NORM     = CFG["ttf_norm_ticks"]
MAX_WIN      = CFG["max_windows_per_split"]
HORIZON_TICKS = [float(x) for x in CFG.get("horizon_ticks", [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0])]
CAUSE_NEAR_WINDOW_TICKS = float(CFG.get("cause_near_window_ticks", 120.0))
TTF_NEAR_WINDOW_TICKS = float(CFG.get("ttf_near_window_ticks", 60.0))
TTF_REFINE_WINDOW_TICKS = float(CFG.get("ttf_refine_window_ticks", 120.0))
HORIZON_THRESHOLD = float(CFG.get("horizon_threshold", 0.5))
TTF_BLEND_HORIZON_IDX = int(np.argmin(np.abs(np.asarray(HORIZON_TICKS) - TTF_REFINE_WINDOW_TICKS)))

TTF_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0]
TTF_BIN_NAMES = [
    "0_to_5", "5_to_10", "10_to_15", "15_to_20", "20_to_30", "30_to_45",
    "45_to_60", "60_to_90", "90_to_120", "120_to_180", "180_to_240",
    "gt_240", "no_failure_or_not_applicable",
]
TTF_FAR_BIN_IDX = len(TTF_BIN_NAMES) - 2
TTF_NO_FAILURE_BIN_IDX = len(TTF_BIN_NAMES) - 1
TTF_REG_BIN_EDGES = [30.0, 60.0, 120.0, 240.0]


def ttf_to_bin_idx(ttf_ticks, y_fail):
    if int(y_fail) != 1:
        return TTF_NO_FAILURE_BIN_IDX
    if pd.isna(ttf_ticks) or float(ttf_ticks) > 240.0:
        return TTF_FAR_BIN_IDX
    t = max(float(ttf_ticks), 0.0)
    for i in range(len(TTF_BIN_EDGES) - 1):
        if TTF_BIN_EDGES[i] <= t <= TTF_BIN_EDGES[i + 1]:
            # Intervals are named 0_to_5, 5_to_10, etc. Exact boundaries go
            # into the shorter warning bin only at zero; otherwise into the
            # upper interval by using t <= upper.
            return i
    return TTF_FAR_BIN_IDX


def ttf_reg_bin_idx(ttf_ticks):
    """Broad positive-only regions used to balance exact-TTF regression."""
    if pd.isna(ttf_ticks):
        return -1
    return int(np.searchsorted(TTF_REG_BIN_EDGES, float(ttf_ticks), side="right"))


# The Graph WaveNet paper uses dilated causal convolutions so the receptive field
# grows exponentially with depth. For kernel size 2, blocks 1, layers 6 gives
# receptive field = 1 + (1 + 2 + 4 + 8 + 16 + 32) = 64 ticks.
TEMPORAL_RECEPTIVE_FIELD = 1 + CFG["blocks"] * sum((CFG["kernel_size"] - 1) * (2 ** i) for i in range(CFG["layers"]))
log(f"temporal receptive field: {TEMPORAL_RECEPTIVE_FIELD} ticks for window_len={WINDOW_LEN}")
if TEMPORAL_RECEPTIVE_FIELD < WINDOW_LEN:
    log("WARNING: receptive field is smaller than the input window; TTF may degrade.")

run_meta_lut = run_meta.set_index("run_id").to_dict(orient="index")


def _window_end_positions(n, g_sorted, y_fail, fail_tick, normal_stride, near_stride):
    """Use stride 5 far from failure and stride 1 inside the 120 tick danger zone."""
    positions = []
    for end_pos in range(WINDOW_LEN - 1, n):
        if y_fail == 1 and not pd.isna(fail_tick):
            end_tick = float(g_sorted.loc[end_pos, "canonical_tick"])
            if end_tick >= float(fail_tick):
                continue
            ttf_ticks = float(fail_tick) - end_tick
            stride_here = near_stride if ttf_ticks <= NEAR_FAILURE_STRIDE_WINDOW_TICKS else normal_stride
        else:
            stride_here = normal_stride
        if (end_pos - (WINDOW_LEN - 1)) % int(stride_here) == 0:
            positions.append(end_pos)
    return positions


def make_window_index(df, lut, window_len, stride):
    """Build supervised windows.

    v11 timing design:
      1. dense horizon labels at 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240 ticks;
      2. TTF bin labels for timing as classification;
      3. global exact TTF is trained on every positive pre-failure window;
      4. a refinement head specializes inside 120 ticks while the reported
         near-failure region remains independently configurable (default 60);
      5. cause and hierarchical cause labels are trained inside 120 ticks;
      6. near-failure windows use stride 1, far windows keep stride 5.
    """
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        if run_id not in lut:
            continue
        meta  = lut[run_id]
        split = meta.get("split", "unknown")
        if split not in {"train", "val", "test"}:
            continue
        g_sorted = g.sort_values("canonical_tick").reset_index(drop=True)
        n = len(g_sorted)
        if n < window_len:
            continue
        fail_tick = meta.get("funnel_failure_tick", np.nan)
        y_fail    = int(meta.get("y_fail_run", 0))
        cause_idx = int(meta.get("cause_idx", 0))
        family_idx = int(meta.get("family_idx", 0))
        effusion_subtype_idx = int(meta.get("effusion_subtype_idx", 0))
        cause_ttf_head_idx = int(meta.get("cause_ttf_head_idx", -1))
        coarse    = meta.get("coarse_cause", "no_failure")

        end_positions = _window_end_positions(n, g_sorted, y_fail, fail_tick, stride, NEAR_FAILURE_STRIDE)
        for end_pos in end_positions:
            end_tick = float(g_sorted.loc[end_pos, "canonical_tick"])
            if y_fail == 1:
                if pd.isna(fail_tick) or end_tick >= float(fail_tick):
                    continue
                ttf_ticks = max(float(fail_tick) - end_tick, 0.0)
                y_ttf = ttf_ticks / float(TTF_NORM)
                has_exact_ttf = int(ttf_ticks <= TTF_NEAR_WINDOW_TICKS)
                has_refine_ttf = int(ttf_ticks <= TTF_REFINE_WINDOW_TICKS)
                has_ttf = 1
                has_cause = int(ttf_ticks <= CAUSE_NEAR_WINDOW_TICKS)
                has_family = has_cause
                has_effusion_subtype = int(has_cause and coarse in {"effusion_underflux", "effusion_undertemp"})
                has_cause_specific_ttf = int(has_refine_ttf and cause_ttf_head_idx >= 0)
                horizon_labels = [1.0 if ttf_ticks <= h else 0.0 for h in HORIZON_TICKS]
            else:
                ttf_ticks = np.nan
                y_ttf = 0.0
                has_exact_ttf = 0
                has_refine_ttf = 0
                has_ttf = 0
                has_cause = 0
                has_family = 0
                has_effusion_subtype = 0
                has_cause_specific_ttf = 0
                horizon_labels = [0.0 for _ in HORIZON_TICKS]

            ttf_bin_idx = ttf_to_bin_idx(ttf_ticks, y_fail)
            start_pos = end_pos - window_len + 1
            row = dict(
                run_id=run_id,
                config_key=meta.get("config_key"),
                split=split,
                start_pos=int(start_pos),
                end_pos=int(end_pos),
                end_tick=end_tick,
                failure_tick=fail_tick,
                y_fail=y_fail,
                cause_idx=cause_idx,
                family_idx=family_idx,
                effusion_subtype_idx=effusion_subtype_idx,
                cause_ttf_head_idx=cause_ttf_head_idx,
                coarse_cause=coarse,
                ttf_ticks=ttf_ticks,
                y_ttf=y_ttf,
                ttf_bin_idx=ttf_bin_idx,
                ttf_reg_bin_idx=ttf_reg_bin_idx(ttf_ticks),
                has_ttf=has_ttf,
                has_exact_ttf=has_exact_ttf,
                has_refine_ttf=has_refine_ttf,
                has_cause=has_cause,
                has_family=has_family,
                has_effusion_subtype=has_effusion_subtype,
                has_cause_specific_ttf=has_cause_specific_ttf,
                has_horizon=1,
                stride_used=int(NEAR_FAILURE_STRIDE if (y_fail == 1 and not pd.isna(ttf_ticks) and ttf_ticks <= NEAR_FAILURE_STRIDE_WINDOW_TICKS) else WINDOW_STRIDE),
            )
            for hi, h in enumerate(HORIZON_TICKS):
                row[f"horizon_{int(h)}"] = horizon_labels[hi]
            rows.append(row)

    out = pd.DataFrame(rows)
    if MAX_WIN and len(out) > 0:
        rng = np.random.default_rng(SEED)
        parts = []
        for s, part in out.groupby("split", sort=False):
            if len(part) > MAX_WIN:
                idx = rng.choice(part.index.to_numpy(), size=MAX_WIN, replace=False)
                parts.append(part.loc[idx])
            else:
                parts.append(part)
        out = pd.concat(parts, ignore_index=True)
    return out.reset_index(drop=True)


log("building window index ...")
window_index_df = make_window_index(df, run_meta_lut, WINDOW_LEN, WINDOW_STRIDE)

# Class weights for the near-failure flat cause head.
_train_cause = window_index_df[
    (window_index_df["split"].eq("train"))
    & (window_index_df["y_fail"].eq(1))
    & (window_index_df["has_cause"].eq(1))
]
if _train_cause.empty:
    log("WARNING: no near-failure cause windows found; falling back to all positive train windows for cause weights")
    _train_cause = window_index_df[(window_index_df["split"].eq("train")) & (window_index_df["y_fail"].eq(1))]
_cause_counts = _train_cause["cause_idx"].value_counts().sort_index()
_all_cause_idx = list(range(len(cause_names)))
_cause_counts = _cause_counts.reindex(_all_cause_idx, fill_value=1)
_cause_weights_np = max(len(_train_cause), 1) / (len(cause_names) * _cause_counts.values.astype(float))
CAUSE_CLASS_WEIGHTS = torch.tensor(_cause_weights_np, dtype=torch.float32)

_train_family = window_index_df[(window_index_df["split"].eq("train")) & (window_index_df["has_family"].eq(1))]
_family_counts = _train_family["family_idx"].value_counts().reindex(range(len(family_names)), fill_value=1)
FAMILY_CLASS_WEIGHTS = torch.tensor(max(len(_train_family), 1) / (len(family_names) * _family_counts.values.astype(float)), dtype=torch.float32)

_train_ttf_bins = window_index_df[window_index_df["split"].eq("train")]
_ttf_bin_counts = _train_ttf_bins["ttf_bin_idx"].value_counts().reindex(range(len(TTF_BIN_NAMES)), fill_value=1)
_ttf_bin_weights_np = np.sqrt(
    max(len(_train_ttf_bins), 1) / (len(TTF_BIN_NAMES) * _ttf_bin_counts.values.astype(float))
)
_ttf_bin_weights_np = np.clip(_ttf_bin_weights_np, 0.25, 4.0)
_ttf_bin_weights_np /= max(float(_ttf_bin_weights_np.mean()), 1e-8)
TTF_BIN_CLASS_WEIGHTS = torch.tensor(_ttf_bin_weights_np, dtype=torch.float32)

# Inverse-sqrt weighting across broad positive TTF regions prevents the dense
# stride-1 near windows from drowning out the farther windows that v8 handled
# well. Only training counts are used, so validation/test distributions do not
# influence optimization.
_train_ttf_reg = window_index_df[
    window_index_df["split"].eq("train") & window_index_df["has_ttf"].eq(1)
]
_n_ttf_reg_bins = len(TTF_REG_BIN_EDGES) + 1
_ttf_reg_counts = _train_ttf_reg["ttf_reg_bin_idx"].value_counts().reindex(
    range(_n_ttf_reg_bins), fill_value=1
)
_ttf_reg_weights_np = np.sqrt(
    max(len(_train_ttf_reg), 1) / (_n_ttf_reg_bins * _ttf_reg_counts.values.astype(float))
)
_ttf_reg_weights_np = np.clip(_ttf_reg_weights_np, 0.50, 3.0)
_ttf_reg_weights_np /= max(float(_ttf_reg_weights_np.mean()), 1e-8)
TTF_REG_WEIGHTS = torch.tensor(_ttf_reg_weights_np, dtype=torch.float32)
window_index_df["ttf_reg_weight"] = window_index_df["ttf_reg_bin_idx"].map(
    {i: float(w) for i, w in enumerate(_ttf_reg_weights_np)}
).fillna(0.0).astype("float32")

log("near-failure cause class weights (higher = rarer class):")
for i, w in enumerate(CAUSE_CLASS_WEIGHTS.tolist()):
    log(f"  {cause_names[i]:30s}  weight={w:.3f}")
log("TTF bin class weights:")
for i, w in enumerate(TTF_BIN_CLASS_WEIGHTS.tolist()):
    log(f"  {TTF_BIN_NAMES[i]:30s}  weight={w:.3f}")
log(f"TTF regression region weights: {TTF_REG_WEIGHTS.tolist()}")

if window_index_df.empty:
    raise ValueError("No windows created -- check WINDOW_LEN vs canonical_tick range")

bad = window_index_df[
    window_index_df["y_fail"].eq(1)
    & pd.to_numeric(window_index_df["end_tick"], errors="coerce").ge(pd.to_numeric(window_index_df["failure_tick"], errors="coerce"))
]
assert len(bad) == 0, "leakage: windows ending at/after failure tick"

log("window counts:")
for s in ["train", "val", "test"]:
    w  = window_index_df[window_index_df["split"].eq(s)]
    log(
        f"  {s:6s}  windows={len(w)}  positives={int(w['y_fail'].sum())}  "
        f"near_cause={int(w['has_cause'].sum())} exact_ttf={int(w['has_exact_ttf'].sum())} "
        f"stride1={int((w['stride_used'] == 1).sum())}"
    )
log("cause windows near failure:")
if "has_cause" in window_index_df.columns and window_index_df["has_cause"].sum() > 0:
    log(window_index_df[window_index_df["has_cause"].eq(1)].groupby(["split", "coarse_cause"]).size().to_string())
else:
    log("no near-failure cause windows")

# =============================================================================
# Model spec + normalization + dataset
# =============================================================================
@dataclass
class ModelSpec:
    name: str
    node_cols_by_node: Dict[str, List[str]]
    context_cols: List[str]
    max_node_features: int
    node_feature_names_by_dim: Dict[str, List[str]]
    context_feature_names: List[str]

def build_spec(name, node_cols, ctx_cols):
    max_f = max(max(len(c), 1) for c in node_cols.values())
    names = {}
    for node in PHYSICAL_NODES:
        cols = list(node_cols.get(node, []))
        names[node] = cols + [f"__pad_{i}" for i in range(max_f - len(cols))]
    return ModelSpec(name=name, node_cols_by_node={k: list(v) for k, v in node_cols.items()},
                     context_cols=list(ctx_cols), max_node_features=max_f,
                     node_feature_names_by_dim=names, context_feature_names=list(ctx_cols))

MODEL_A_SPEC = build_spec("subsystem_only",            node_cols_by_node, [])
MODEL_B_SPEC = build_spec("subsystem_plus_safe_context", node_cols_by_node, context_cols_b)

all_feat_cols = set()
for cols in node_cols_by_node.values():
    all_feat_cols.update(cols)
all_feat_cols.update(context_cols_b)

keep_cols  = ["run_id","canonical_tick"] + sorted(all_feat_cols)
keep_cols  = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
df_model   = df[keep_cols].copy()

for col in sorted(all_feat_cols):
    if col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").astype(np.float32)

df_model["run_id"]         = df_model["run_id"].astype(str)
df_model["canonical_tick"] = pd.to_numeric(df_model["canonical_tick"], errors="coerce")

run_frames = {rid: g.sort_values("canonical_tick").reset_index(drop=True)
              for rid, g in df_model.groupby("run_id", sort=False)}

del features_df, df, df_model
gc.collect()

train_runs = set(window_index_df.loc[window_index_df["split"].eq("train"), "run_id"].astype(str))

def compute_normalizer(col):
    count = total = sq = 0.0
    for rid in train_runs:
        if rid not in run_frames or col not in run_frames[rid].columns:
            continue
        v = pd.to_numeric(run_frames[rid][col], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        if not m.any(): continue
        c = v[m]; count += c.size; total += float(c.sum()); sq += float(np.square(c).sum())
    if count == 0: return 0.0, 1.0
    mean = total / count
    var  = max((sq - total*total/count) / (count-1) if count > 1 else 0.0, 0.0)
    std  = math.sqrt(var)
    return float(mean), float(std) if (np.isfinite(std) and std >= 1e-8) else 1.0

log("computing normalizer …")
normalizer = {col: compute_normalizer(col) for col in sorted(all_feat_cols)}

def normalize_array(values, col):
    mean, std = normalizer.get(col, (0.0, 1.0))
    out = (values.astype(np.float32) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

class SpaceForgeWindowDataset(Dataset):
    """Window dataset with dense horizons, TTF bins, hierarchical causes, and exact TTF masks."""
    def __init__(self, window_index, run_frames, spec):
        self.wi       = window_index.reset_index(drop=True).copy()
        self.spec     = spec
        self.run_ids  = self.wi["run_id"].astype(str).to_numpy()
        self.starts   = self.wi["start_pos"].astype(int).to_numpy()
        self.ends     = self.wi["end_pos"].astype(int).to_numpy()
        self.end_ticks= self.wi["end_tick"].astype(float).to_numpy()
        self.y_fail   = self.wi["y_fail"].astype(float).to_numpy()
        self.cause_idx= self.wi["cause_idx"].astype(int).to_numpy()
        self.family_idx = self.wi["family_idx"].astype(int).to_numpy()
        self.effusion_subtype_idx = self.wi["effusion_subtype_idx"].astype(int).to_numpy()
        self.cause_ttf_head_idx = self.wi["cause_ttf_head_idx"].astype(int).to_numpy()
        self.ttf_bin_idx = self.wi["ttf_bin_idx"].astype(int).to_numpy()
        self.y_ttf    = self.wi["y_ttf"].astype(float).to_numpy()
        self.ttf_ticks= pd.to_numeric(self.wi.get("ttf_ticks", np.nan), errors="coerce").to_numpy(dtype=np.float32)
        self.ttf_reg_weight = self.wi["ttf_reg_weight"].astype(float).to_numpy(dtype=np.float32)
        self.has_ttf  = self.wi["has_ttf"].astype(float).to_numpy()
        self.has_exact_ttf = self.wi["has_exact_ttf"].astype(float).to_numpy()
        self.has_refine_ttf = self.wi["has_refine_ttf"].astype(float).to_numpy()
        self.has_cause= self.wi["has_cause"].astype(float).to_numpy()
        self.has_family = self.wi["has_family"].astype(float).to_numpy()
        self.has_effusion_subtype = self.wi["has_effusion_subtype"].astype(float).to_numpy()
        self.has_cause_specific_ttf = self.wi["has_cause_specific_ttf"].astype(float).to_numpy()
        hcols = [f"horizon_{int(h)}" for h in HORIZON_TICKS]
        self.horizon_labels = self.wi[hcols].astype(float).to_numpy(dtype=np.float32)
        self.cache    = {}
        for run_id in sorted(set(self.run_ids)):
            g     = run_frames[run_id]
            n_n   = len(PHYSICAL_NODES)
            max_f = spec.max_node_features
            t_len = len(g)
            x = np.zeros((n_n, max_f, t_len), dtype=np.float32)
            for ni, node in enumerate(PHYSICAL_NODES):
                for fi, col in enumerate(spec.node_cols_by_node.get(node, [])[:max_f]):
                    if col in g.columns:
                        x[ni, fi, :] = normalize_array(pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32), col)
            c_cols = spec.context_cols
            ctx = np.zeros((len(c_cols), t_len), dtype=np.float32) if c_cols else np.zeros((0, t_len), dtype=np.float32)
            for ci, col in enumerate(c_cols):
                if col in g.columns:
                    ctx[ci, :] = normalize_array(pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32), col)
            self.cache[run_id] = {"x": torch.from_numpy(x), "context": torch.from_numpy(ctx)}
        gc.collect()

    def __len__(self):
        return len(self.wi)

    def __getitem__(self, idx):
        rid   = self.run_ids[idx]
        s, e  = self.starts[idx], self.ends[idx]
        cached= self.cache[rid]
        return {
            "run_id":    rid,
            "x":        cached["x"][:, :, s:e+1].contiguous(),
            "context":  cached["context"][:, s:e+1].contiguous(),
            "y_fail":   torch.tensor(self.y_fail[idx], dtype=torch.float32),
            "cause_idx":torch.tensor(self.cause_idx[idx], dtype=torch.long),
            "family_idx": torch.tensor(self.family_idx[idx], dtype=torch.long),
            "effusion_subtype_idx": torch.tensor(self.effusion_subtype_idx[idx], dtype=torch.long),
            "cause_ttf_head_idx": torch.tensor(self.cause_ttf_head_idx[idx], dtype=torch.long),
            "ttf_bin_idx": torch.tensor(self.ttf_bin_idx[idx], dtype=torch.long),
            "y_ttf":    torch.tensor(self.y_ttf[idx], dtype=torch.float32),
            "ttf_ticks":torch.tensor(np.nan_to_num(self.ttf_ticks[idx], nan=-1.0), dtype=torch.float32),
            "ttf_reg_weight": torch.tensor(self.ttf_reg_weight[idx], dtype=torch.float32),
            "has_ttf":  torch.tensor(self.has_ttf[idx], dtype=torch.float32),
            "has_exact_ttf": torch.tensor(self.has_exact_ttf[idx], dtype=torch.float32),
            "has_refine_ttf": torch.tensor(self.has_refine_ttf[idx], dtype=torch.float32),
            "has_cause":torch.tensor(self.has_cause[idx], dtype=torch.float32),
            "has_family": torch.tensor(self.has_family[idx], dtype=torch.float32),
            "has_effusion_subtype": torch.tensor(self.has_effusion_subtype[idx], dtype=torch.float32),
            "has_cause_specific_ttf": torch.tensor(self.has_cause_specific_ttf[idx], dtype=torch.float32),
            "horizon_labels": torch.tensor(self.horizon_labels[idx], dtype=torch.float32),
            "window_id":torch.tensor(idx, dtype=torch.long),
            "end_tick": torch.tensor(self.end_ticks[idx], dtype=torch.float32),
        }


def make_loaders(spec, batch_size):
    ds, loaders = {}, {}
    for split in ["train", "val", "test"]:
        wi = window_index_df[window_index_df["split"].eq(split)].reset_index(drop=True)
        log(f"building {spec.name} {split} dataset: {len(wi)} windows")
        ds[split] = SpaceForgeWindowDataset(wi, run_frames, spec)
        if split == "train":
            # Balance positive TTF regions while retaining a bounded rare-cause
            # boost. Near windows are already dense because they use stride 1.
            _wi_causes = wi["cause_idx"].astype(int).values
            _sample_w  = np.ones(len(wi), dtype=np.float64)
            _near = wi["has_cause"].astype(int).values == 1
            for i, c in enumerate(_wi_causes):
                if wi.loc[i, "y_fail"] == 1:
                    _sample_w[i] = 1.5 * max(float(wi.loc[i, "ttf_reg_weight"]), 0.5)
                if _near[i]:
                    _sample_w[i] *= min(max(float(CAUSE_CLASS_WEIGHTS[c]), 1.0), 3.0) ** 0.5
            _sampler = WeightedRandomSampler(
                weights=torch.from_numpy(_sample_w),
                num_samples=len(_sample_w),
                replacement=True,
            )
            loaders[split] = DataLoader(
                ds[split], batch_size=batch_size, sampler=_sampler,
                num_workers=0, pin_memory=torch.cuda.is_available())
        else:
            loaders[split] = DataLoader(
                ds[split], batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available())
    return ds, loaders

# =============================================================================
# Graph supports
# =============================================================================
PHYSICAL_EDGES = [
    # Orbital environment drives illumination and thermal exposure.
    ("orbit", "solar_array"),
    ("orbit", "array_gimbal"),
    ("orbit", "battery"),
    ("orbit", "battery_thermal"),
    ("orbit", "effusion_cell"),
    ("orbit", "substrate"),
    ("orbit", "cryo_panel"),
    ("orbit", "radiator"),

    # Pointing controls generation; generation/storage feed the main bus.
    ("array_gimbal", "solar_array"),
    ("solar_array", "power_bus"),
    ("battery", "power_bus"),
    ("simulation_engine", "battery"),
    ("simulation_engine", "power_bus"),

    # Electrical loads and actuators draw from the bus.
    ("power_bus", "array_gimbal"),
    ("power_bus", "battery_thermal"),
    ("power_bus", "heater_bank"),
    ("power_bus", "cryo_panel"),
    ("power_bus", "radiator"),
    ("heater_bank", "effusion_cell"),
    ("heater_bank", "substrate"),

    # Battery temperature and electrical limits are mutually coupled.
    ("battery", "battery_thermal"),
    ("battery_thermal", "battery"),

    # The source inventory changes effusion thermal response and is depleted
    # by the cell's current process state.
    ("effusion_cell", "source_inventory"),
    ("source_inventory", "effusion_cell"),

    # Thermal loads reject heat into the shared radiator loop.
    ("battery_thermal", "radiator"),
    ("cryo_panel", "radiator"),
    ("radiator", "battery_thermal"),

    # Material flux from the effusion cell ultimately affects the substrate.
    ("effusion_cell", "substrate"),
]
node_to_idx = {n: i for i, n in enumerate(PHYSICAL_NODES)}
idx_to_node = {i: n for n, i in node_to_idx.items()}

n = len(PHYSICAL_NODES)
adj = np.zeros((n, n), dtype=np.float32)
for src, dst in PHYSICAL_EDGES:
    adj[node_to_idx[src], node_to_idx[dst]] = 1.0

def row_norm(mat):
    mat = mat.astype(np.float32)
    d   = mat.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return mat / d

fixed_supports = [torch.tensor(row_norm(adj), dtype=torch.float32),
                  torch.tensor(row_norm(adj.T), dtype=torch.float32)]

# =============================================================================
# Model
# =============================================================================
def nconv(x, support):
    return torch.einsum("bcnt,nm->bcmt", x, support).contiguous()


def mask_entropy(mask, eps=1e-8):
    mask = torch.clamp(mask, eps, 1.0 - eps)
    return -mask * torch.log(mask) - (1.0 - mask) * torch.log(1.0 - mask)


class DiffusionGraphConv(nn.Module):
    def __init__(self, channels, out_channels, dropout, order, num_supports):
        super().__init__()
        self.order = order
        self.num_supports = num_supports
        self.mlp = nn.Conv2d(channels * (1 + order * num_supports), out_channels, kernel_size=(1,1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):
        out = [x]
        for sup in supports:
            x1 = nconv(x, sup)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = nconv(x1, sup)
                out.append(x1)
        return self.dropout(self.mlp(torch.cat(out, dim=1)))


class CausalConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), dilation=(1, dilation))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0, 0, 0)))


def dual_range_ttf_predictions(global_raw, near_raw, horizon_logits):
    """Blend an all-range estimate with a near-failure specialist.

    The gate is the model's own P(failure within the refinement window). It is
    detached here so regression cannot sacrifice horizon calibration merely to
    choose an easier expert.
    """
    global_pred = F.softplus(global_raw).squeeze(-1)
    near_pred = F.softplus(near_raw).squeeze(-1)
    near_gate = torch.sigmoid(horizon_logits[:, TTF_BLEND_HORIZON_IDX]).detach()
    blended = (1.0 - near_gate) * global_pred + near_gate * near_pred
    return global_pred, near_pred, blended, near_gate


class SpaceForgeGraphWaveNet(nn.Module):
    """Graph WaveNet backbone with v11 dual-range timing heads.

    The backbone remains faithful to Graph WaveNet: forward/backward directed
    diffusion supports, learned adaptive adjacency, gated dilated causal temporal
    convolutions, graph convolution at each temporal scale, residual connections,
    and skip connections. The global TTF expert learns the full countdown while
    the near expert resolves the final approach to failure.
    """
    def __init__(self, num_nodes, in_dim, num_causes, fixed_supports,
                 context_dim=0, num_horizons=11, num_ttf_bins=12,
                 num_families=6, num_effusion_subtypes=3, num_cause_ttf_heads=5,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=64, end_channels=96, kernel_size=2,
                 blocks=1, layers=6, dropout=0.20, graph_order=2,
                 adaptive_adj=True, node_embedding_dim=10, context_dropout=0.50):
        super().__init__()
        self.num_nodes   = num_nodes
        self.in_dim      = in_dim
        self.num_causes  = num_causes
        self.num_horizons= num_horizons
        self.num_ttf_bins = num_ttf_bins
        self.num_families = num_families
        self.num_effusion_subtypes = num_effusion_subtypes
        self.num_cause_ttf_heads = num_cause_ttf_heads
        self.context_dim = context_dim
        self.dropout     = dropout
        self.graph_order = graph_order
        self.adaptive_adj= adaptive_adj

        self.register_buffer("fixed_support_0", fixed_supports[0].clone().float())
        self.register_buffer("fixed_support_1", fixed_supports[1].clone().float())
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1,1))

        support_count = 2
        if adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(node_embedding_dim, num_nodes) * 0.1)
            support_count += 1

        fc, gc_, rc, sc, grc, bn = [], [], [], [], [], []
        for _ in range(blocks):
            for i in range(layers):
                d = 2 ** i
                fc.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, d))
                gc_.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, d))
                rc.append(nn.Conv2d(dilation_channels, residual_channels, (1,1)))
                sc.append(nn.Conv2d(dilation_channels, skip_channels, (1,1)))
                grc.append(DiffusionGraphConv(dilation_channels, residual_channels, dropout, graph_order, support_count))
                bn.append(nn.BatchNorm2d(residual_channels))
        self.filter_convs   = nn.ModuleList(fc)
        self.gate_convs     = nn.ModuleList(gc_)
        self.residual_convs = nn.ModuleList(rc)
        self.skip_convs     = nn.ModuleList(sc)
        self.graph_convs    = nn.ModuleList(grc)
        self.batch_norms    = nn.ModuleList(bn)

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1,1))
        self.end_conv_2 = nn.Conv2d(end_channels,  end_channels, (1,1))

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Conv1d(context_dim, end_channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(context_dropout),
                nn.Conv1d(end_channels, end_channels, kernel_size=3, padding=1), nn.ReLU(),
            )
            head_in = end_channels * 2
        else:
            self.context_encoder = None
            head_in = end_channels

        self.binary_head = nn.Linear(head_in, 1)
        self.cause_head  = nn.Linear(head_in, num_causes)
        self.family_head = nn.Linear(head_in, num_families)
        self.effusion_subtype_head = nn.Linear(head_in, num_effusion_subtypes)
        self.ttf_global_head = nn.Linear(head_in, 1)
        self.ttf_near_head = nn.Linear(head_in, 1)
        self.cause_specific_ttf_head = nn.Linear(head_in, num_cause_ttf_heads)
        self.ttf_bin_head = nn.Linear(head_in, num_ttf_bins)
        self.horizon_head = nn.Linear(head_in, num_horizons)

    def base_supports(self):
        supports = [self.fixed_support_0, self.fixed_support_1]
        if self.adaptive_adj:
            supports.append(torch.softmax(torch.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1))
        return supports

    def apply_masks_to_supports(self, supports, support_masks=None):
        if support_masks is None:
            return supports
        masked = []
        for i, sup in enumerate(supports):
            if i < len(support_masks) and support_masks[i] is not None:
                m = support_masks[i].to(sup.device)
                sm = sup * m
                sm = sm / sm.sum(dim=1, keepdim=True).clamp_min(1e-8)
                masked.append(sm)
            else:
                masked.append(sup)
        return masked

    def forward(self, x, context=None, support_masks=None, feature_mask=None,
                context_mask=None, temporal_mask=None, context_override=None):
        if feature_mask is not None:
            fm = feature_mask.to(x.device)
            x = x * (fm.view(1,1,-1,1) if fm.dim()==1 else fm.view(1,fm.shape[0],fm.shape[1],1))
        if temporal_mask is not None:
            x = x * temporal_mask.to(x.device).view(1,1,1,-1)
        if context is not None and context.numel() > 0:
            if context_override is not None:
                context = context_override.to(context.device)
            if context_mask is not None:
                context = context * context_mask.to(context.device).view(1,-1,1)
            if temporal_mask is not None:
                context = context * temporal_mask.to(context.device).view(1,1,-1)

        # Input is [batch, nodes, features, time]. GraphWaveNet convs expect
        # [batch, channels/features, nodes, time].
        x = self.start_conv(x.permute(0,2,1,3).contiguous())
        supports = self.apply_masks_to_supports(self.base_supports(), support_masks)
        skip = None
        for fc, gc_, rc, sc, grc, bn in zip(
            self.filter_convs, self.gate_convs, self.residual_convs,
            self.skip_convs, self.graph_convs, self.batch_norms
        ):
            res  = x
            x    = torch.tanh(fc(x)) * torch.sigmoid(gc_(x))
            s    = sc(x)
            skip = s if skip is None else skip[..., -s.size(3):] + s
            x    = rc(grc(x, supports))
            x    = bn(x + res[..., -x.size(3):])

        h_map = F.relu(self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip)))))
        h = h_map[:, :, :, -1].mean(dim=2)
        if self.context_encoder is not None and context is not None and context.numel() > 0:
            ctx_h = self.context_encoder(context)[:, :, -1]
            h = torch.cat([h, ctx_h], dim=1)
        horizon_logits = self.horizon_head(h)
        ttf_global, ttf_near, ttf_blended, ttf_gate = dual_range_ttf_predictions(
            self.ttf_global_head(h), self.ttf_near_head(h), horizon_logits
        )
        return {
            "binary_logit": self.binary_head(h).squeeze(-1),
            "cause_logits": self.cause_head(h),
            "family_logits": self.family_head(h),
            "effusion_subtype_logits": self.effusion_subtype_head(h),
            "ttf_pred": ttf_blended,
            "ttf_global_pred": ttf_global,
            "ttf_near_pred": ttf_near,
            "ttf_near_gate": ttf_gate,
            "cause_specific_ttf_pred": F.softplus(self.cause_specific_ttf_head(h)),
            "ttf_bin_logits": self.ttf_bin_head(h),
            "horizon_logits": horizon_logits,
        }


class NodeLSTMModel(nn.Module):
    """Per-node LSTM baseline. Temporal but no graph diffusion."""
    def __init__(self, num_nodes, in_dim, num_causes, hidden_dim=64,
                 num_layers=2, context_dim=0, num_horizons=11,
                 num_ttf_bins=12, num_families=6, num_effusion_subtypes=3,
                 num_cause_ttf_heads=5, context_dropout=0.50, dropout=0.20):
        super().__init__()
        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        head_in = num_nodes * hidden_dim
        if context_dim > 0:
            self.ctx_enc = nn.Sequential(
                nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Dropout(context_dropout))
            head_in += hidden_dim
        else:
            self.ctx_enc = None
        self.binary_head = nn.Linear(head_in, 1)
        self.cause_head  = nn.Linear(head_in, num_causes)
        self.family_head = nn.Linear(head_in, num_families)
        self.effusion_subtype_head = nn.Linear(head_in, num_effusion_subtypes)
        self.ttf_global_head = nn.Linear(head_in, 1)
        self.ttf_near_head = nn.Linear(head_in, 1)
        self.cause_specific_ttf_head = nn.Linear(head_in, num_cause_ttf_heads)
        self.ttf_bin_head = nn.Linear(head_in, num_ttf_bins)
        self.horizon_head= nn.Linear(head_in, num_horizons)

    def forward(self, x, context=None, **kwargs):
        B, N, n_feat, T = x.shape
        xr = x.permute(0, 1, 3, 2).reshape(B * N, T, n_feat)
        _, (h, _) = self.lstm(xr)
        feat = h[-1].reshape(B, N * self.hidden_dim)
        if self.ctx_enc is not None and context is not None and context.numel() > 0:
            feat = torch.cat([feat, self.ctx_enc(context[:, :, -1])], dim=1)
        horizon_logits = self.horizon_head(feat)
        ttf_global, ttf_near, ttf_blended, ttf_gate = dual_range_ttf_predictions(
            self.ttf_global_head(feat), self.ttf_near_head(feat), horizon_logits
        )
        return {
            "binary_logit": self.binary_head(feat).squeeze(-1),
            "cause_logits": self.cause_head(feat),
            "family_logits": self.family_head(feat),
            "effusion_subtype_logits": self.effusion_subtype_head(feat),
            "ttf_pred": ttf_blended,
            "ttf_global_pred": ttf_global,
            "ttf_near_pred": ttf_near,
            "ttf_near_gate": ttf_gate,
            "cause_specific_ttf_pred": F.softplus(self.cause_specific_ttf_head(feat)),
            "ttf_bin_logits": self.ttf_bin_head(feat),
            "horizon_logits": horizon_logits,
        }


class FlatMLPModel(nn.Module):
    """Flatten baseline. No explicit temporal or graph inductive bias."""
    def __init__(self, num_nodes, in_dim, window_len, num_causes,
                 hidden_dim=256, context_dim=0, num_horizons=11,
                 num_ttf_bins=12, num_families=6, num_effusion_subtypes=3,
                 num_cause_ttf_heads=5, context_dropout=0.50, dropout=0.20):
        super().__init__()
        flat_dim = num_nodes * in_dim * window_len
        self.ctx_flat = context_dim * window_len if context_dim > 0 else 0
        flat_dim += self.ctx_flat
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        out_dim = hidden_dim // 2
        self.binary_head = nn.Linear(out_dim, 1)
        self.cause_head  = nn.Linear(out_dim, num_causes)
        self.family_head = nn.Linear(out_dim, num_families)
        self.effusion_subtype_head = nn.Linear(out_dim, num_effusion_subtypes)
        self.ttf_global_head = nn.Linear(out_dim, 1)
        self.ttf_near_head = nn.Linear(out_dim, 1)
        self.cause_specific_ttf_head = nn.Linear(out_dim, num_cause_ttf_heads)
        self.ttf_bin_head = nn.Linear(out_dim, num_ttf_bins)
        self.horizon_head= nn.Linear(out_dim, num_horizons)

    def forward(self, x, context=None, **kwargs):
        B = x.shape[0]
        parts = [x.reshape(B, -1)]
        if self.ctx_flat > 0 and context is not None and context.numel() > 0:
            parts.append(context.reshape(B, -1))
        h = self.net(torch.cat(parts, dim=1))
        horizon_logits = self.horizon_head(h)
        ttf_global, ttf_near, ttf_blended, ttf_gate = dual_range_ttf_predictions(
            self.ttf_global_head(h), self.ttf_near_head(h), horizon_logits
        )
        return {
            "binary_logit": self.binary_head(h).squeeze(-1),
            "cause_logits": self.cause_head(h),
            "family_logits": self.family_head(h),
            "effusion_subtype_logits": self.effusion_subtype_head(h),
            "ttf_pred": ttf_blended,
            "ttf_global_pred": ttf_global,
            "ttf_near_pred": ttf_near,
            "ttf_near_gate": ttf_gate,
            "cause_specific_ttf_pred": F.softplus(self.cause_specific_ttf_head(h)),
            "ttf_bin_logits": self.ttf_bin_head(h),
            "horizon_logits": horizon_logits,
        }

# =============================================================================
# Training utilities
# =============================================================================
USE_TTF = CFG["use_ttf"]
USE_AMP = DEVICE.type == "cuda"


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def autocast_ctx():
    if not USE_AMP:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=DEVICE.type, enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def to_device(batch):
    return {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def monotonic_horizon_penalty(horizon_logits):
    """Penalize violations of P(5) <= P(10) <= ... <= P(240)."""
    probs = torch.sigmoid(horizon_logits)
    return F.relu(probs[:, :-1] - probs[:, 1:]).mean()


def multitask_loss(outputs, batch, pos_weight=None):
    """Joint v11 loss with balanced global and near-failure TTF objectives."""
    y_fail   = batch["y_fail"].float()
    cause    = batch["cause_idx"].long()
    family   = batch["family_idx"].long()
    subtype  = batch["effusion_subtype_idx"].long()
    ttf_bin  = batch["ttf_bin_idx"].long()
    y_ttf    = batch["y_ttf"].float()
    horizons = batch["horizon_labels"].float()

    has_cause = batch["has_cause"].float()
    has_family = batch["has_family"].float()
    has_effusion_subtype = batch["has_effusion_subtype"].float()
    has_ttf = batch["has_ttf"].float()
    has_exact_ttf = batch["has_exact_ttf"].float()
    has_refine_ttf = batch["has_refine_ttf"].float()
    has_cause_specific_ttf = batch["has_cause_specific_ttf"].float()
    ttf_reg_weight = batch["ttf_reg_weight"].float()

    bce = F.binary_cross_entropy_with_logits(outputs["binary_logit"], y_fail, pos_weight=pos_weight)

    cw = CAUSE_CLASS_WEIGHTS.to(outputs["cause_logits"].device)
    cause_mask = y_fail.eq(1.0) & has_cause.eq(1.0)
    if cause_mask.any():
        cause_loss = F.cross_entropy(outputs["cause_logits"][cause_mask], cause[cause_mask], weight=cw)
    else:
        cause_loss = outputs["cause_logits"].sum() * 0.0

    fw = FAMILY_CLASS_WEIGHTS.to(outputs["family_logits"].device)
    family_mask = y_fail.eq(1.0) & has_family.eq(1.0)
    if family_mask.any():
        family_loss = F.cross_entropy(outputs["family_logits"][family_mask], family[family_mask], weight=fw)
    else:
        family_loss = outputs["family_logits"].sum() * 0.0

    subtype_mask = y_fail.eq(1.0) & has_effusion_subtype.eq(1.0)
    if subtype_mask.any():
        subtype_loss = F.cross_entropy(outputs["effusion_subtype_logits"][subtype_mask], subtype[subtype_mask])
    else:
        subtype_loss = outputs["effusion_subtype_logits"].sum() * 0.0

    ttf_bin_weights = TTF_BIN_CLASS_WEIGHTS.to(outputs["ttf_bin_logits"].device)
    ttf_bin_loss = F.cross_entropy(outputs["ttf_bin_logits"], ttf_bin, weight=ttf_bin_weights)

    horizon_loss = F.binary_cross_entropy_with_logits(outputs["horizon_logits"], horizons)
    mono_loss = monotonic_horizon_penalty(outputs["horizon_logits"])

    all_ttf_mask = y_fail.eq(1.0) & has_ttf.eq(1.0)
    refine_mask = y_fail.eq(1.0) & has_refine_ttf.eq(1.0)
    near_boost = 1.0 + (float(CFG.get("near_ttf_loss_multiplier", 2.0)) - 1.0) * has_exact_ttf

    def weighted_ttf_huber(pred, mask, extra_weight=None):
        if not (USE_TTF and mask.any()):
            return pred.sum() * 0.0
        per_item = F.smooth_l1_loss(pred[mask], y_ttf[mask], reduction="none")
        weights = ttf_reg_weight[mask].clamp_min(0.05)
        if extra_weight is not None:
            weights = weights * extra_weight[mask]
        return (per_item * weights).sum() / weights.sum().clamp_min(1e-8)

    if USE_TTF and all_ttf_mask.any():
        # The blended prediction is the deployed output. Its loss emphasizes
        # <=60-tick samples moderately while inverse-region weights preserve
        # useful gradients from the full countdown.
        ttf_loss = weighted_ttf_huber(
            outputs["ttf_pred"], all_ttf_mask, near_boost
        )
        global_ttf_loss = weighted_ttf_huber(
            outputs["ttf_global_pred"], all_ttf_mask
        )
    else:
        ttf_loss = outputs["ttf_pred"].sum() * 0.0
        global_ttf_loss = outputs["ttf_global_pred"].sum() * 0.0

    near_ttf_loss = weighted_ttf_huber(
        outputs["ttf_near_pred"], refine_mask, near_boost
    )

    cst_mask = y_fail.eq(1.0) & has_cause_specific_ttf.eq(1.0)
    if USE_TTF and cst_mask.any():
        head_idx = batch["cause_ttf_head_idx"][cst_mask].long()
        row_idx = torch.arange(head_idx.numel(), device=head_idx.device)
        pred = outputs["cause_specific_ttf_pred"][cst_mask][row_idx, head_idx]
        per_cause = F.smooth_l1_loss(pred, y_ttf[cst_mask], reduction="none")
        cst_weights = near_boost[cst_mask]
        cause_ttf_loss = (per_cause * cst_weights).sum() / cst_weights.sum().clamp_min(1e-8)
    else:
        cause_ttf_loss = outputs["cause_specific_ttf_pred"].sum() * 0.0

    total = (
        bce
        + CFG["cause_loss_weight"] * cause_loss
        + CFG["family_loss_weight"] * family_loss
        + CFG["effusion_subtype_loss_weight"] * subtype_loss
        + CFG["ttf_bin_loss_weight"] * ttf_bin_loss
        + CFG["ttf_loss_weight"] * ttf_loss
        + CFG["global_ttf_loss_weight"] * global_ttf_loss
        + CFG["near_ttf_loss_weight"] * near_ttf_loss
        + CFG["cause_specific_ttf_loss_weight"] * cause_ttf_loss
        + CFG["horizon_loss_weight"] * horizon_loss
        + CFG["monotonic_horizon_loss_weight"] * mono_loss
    )
    return total, {
        "bce": float(bce.detach().cpu()),
        "cause": float(cause_loss.detach().cpu()),
        "family": float(family_loss.detach().cpu()),
        "effusion_subtype": float(subtype_loss.detach().cpu()),
        "ttf_bin": float(ttf_bin_loss.detach().cpu()),
        "ttf": float(ttf_loss.detach().cpu()),
        "ttf_global": float(global_ttf_loss.detach().cpu()),
        "ttf_near": float(near_ttf_loss.detach().cpu()),
        "cause_ttf": float(cause_ttf_loss.detach().cpu()),
        "horizon": float(horizon_loss.detach().cpu()),
        "monotonic": float(mono_loss.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, pos_weight, scaler, name, epoch):
    model.train()
    losses, parts = [], []
    total = len(loader)
    every = max(1, total // 8)
    t0 = time.perf_counter()
    for bi, batch in enumerate(loader, 1):
        batch = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(batch["x"], batch["context"])
            loss, lp = multitask_loss(out, batch, pos_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        parts.append(lp)
        if bi == 1 or bi == total or bi % every == 0:
            log(f"{name} epoch {epoch:03d} batch {bi:05d}/{total:05d} loss={np.mean(losses[-every:]):.4f} elapsed={time.perf_counter()-t0:.1f}s")
    out = {"loss": float(np.mean(losses))}
    if parts:
        out.update({k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
    return out


def estimate_ttf_from_horizons(prob_row, threshold=None):
    threshold = HORIZON_THRESHOLD if threshold is None else threshold
    for h, p in zip(HORIZON_TICKS, prob_row):
        if float(p) >= threshold:
            return float(h)
    return float(max(HORIZON_TICKS) + 1.0)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    rows = []
    for batch in loader:
        bd  = to_device(batch)
        out = model(bd["x"], bd["context"])
        probs        = torch.sigmoid(out["binary_logit"]).detach().cpu().numpy()
        cause_logits = out["cause_logits"]
        cause_pred   = cause_logits.argmax(dim=1).detach().cpu().numpy()
        family_pred  = out["family_logits"].argmax(dim=1).detach().cpu().numpy()
        subtype_pred = out["effusion_subtype_logits"].argmax(dim=1).detach().cpu().numpy()
        ttf_bin_pred = out["ttf_bin_logits"].argmax(dim=1).detach().cpu().numpy()
        ttf_pred     = out["ttf_pred"].detach().cpu().numpy()
        ttf_global_pred = out["ttf_global_pred"].detach().cpu().numpy()
        ttf_near_pred = out["ttf_near_pred"].detach().cpu().numpy()
        ttf_near_gate = out["ttf_near_gate"].detach().cpu().numpy()
        cause_ttf_pred_all = out["cause_specific_ttf_pred"].detach().cpu().numpy()
        horizon_prob = torch.sigmoid(out["horizon_logits"]).detach().cpu().numpy()

        for i in range(len(probs)):
            true_head = int(batch["cause_ttf_head_idx"][i])
            pred_cause_name = idx_to_cause.get(int(cause_pred[i]), "unknown_failure")
            pred_head = cause_ttf_to_idx.get(pred_cause_name, -1)
            cst_true = float(cause_ttf_pred_all[i, true_head]) if true_head >= 0 else np.nan
            cst_pred = float(cause_ttf_pred_all[i, pred_head]) if pred_head >= 0 else np.nan
            hier_cause_pred_name = family_to_hier_cause_name(int(family_pred[i]), int(subtype_pred[i]))
            hier_cause_pred = cause_to_idx.get(hier_cause_pred_name, cause_to_idx.get("unknown_failure", 0))
            row = {
                "run_id": str(batch["run_id"][i]),
                "end_tick": float(batch["end_tick"][i]),
                "prob_fail": float(probs[i]),
                "y_fail": int(batch["y_fail"][i]),
                "cause_idx": int(batch["cause_idx"][i]),
                "cause_pred": int(cause_pred[i]),
                "family_idx": int(batch["family_idx"][i]),
                "family_pred": int(family_pred[i]),
                "effusion_subtype_idx": int(batch["effusion_subtype_idx"][i]),
                "effusion_subtype_pred": int(subtype_pred[i]),
                "hier_cause_pred": int(hier_cause_pred),
                "ttf_bin_idx": int(batch["ttf_bin_idx"][i]),
                "ttf_bin_pred": int(ttf_bin_pred[i]),
                "y_ttf": float(batch["y_ttf"][i]),
                "ttf_ticks": float(batch["ttf_ticks"][i]),
                "has_ttf": int(batch["has_ttf"][i]),
                "has_exact_ttf": int(batch["has_exact_ttf"][i]),
                "has_refine_ttf": int(batch["has_refine_ttf"][i]),
                "has_cause": int(batch["has_cause"][i]),
                "has_family": int(batch["has_family"][i]),
                "has_effusion_subtype": int(batch["has_effusion_subtype"][i]),
                "has_cause_specific_ttf": int(batch["has_cause_specific_ttf"][i]),
                "cause_ttf_head_idx": int(batch["cause_ttf_head_idx"][i]),
                "ttf_pred": float(ttf_pred[i]),
                "ttf_global_pred": float(ttf_global_pred[i]),
                "ttf_near_pred": float(ttf_near_pred[i]),
                "ttf_near_gate": float(ttf_near_gate[i]),
                "cause_ttf_pred_true_cause": cst_true,
                "cause_ttf_pred_pred_cause": cst_pred,
                "horizon_estimated_ttf_ticks": estimate_ttf_from_horizons(horizon_prob[i]),
                "window_id": int(batch["window_id"][i]),
            }
            y_h = batch["horizon_labels"][i].detach().cpu().numpy()
            for hi, h in enumerate(HORIZON_TICKS):
                row[f"horizon_{int(h)}_true"] = float(y_h[hi])
                row[f"horizon_{int(h)}_prob"] = float(horizon_prob[i, hi])
            rows.append(row)
    return pd.DataFrame(rows)


def family_to_hier_cause_name(family_idx, subtype_idx):
    fam = idx_to_family.get(int(family_idx), "unknown_failure")
    if fam == "battery":
        return "battery_power"
    if fam == "effusion":
        subtype = idx_to_effusion_subtype.get(int(subtype_idx), "not_effusion")
        return subtype if subtype in {"effusion_underflux", "effusion_undertemp"} else "effusion_underflux"
    if fam == "substrate":
        return "substrate_undertemp"
    if fam == "stall":
        return "stall"
    if fam == "no_failure":
        return "no_failure"
    return "unknown_failure"


def _binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    y_pred = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(accuracy_score(y_true, y_pred))}
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, probs))
        out["auprc"] = float(average_precision_score(y_true, probs))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def _class_report_dict(y_true, y_pred, labels, names):
    if len(y_true) == 0:
        return {"macro_f1": np.nan, "weighted_f1": np.nan, "per_class_f1": {n: np.nan for n in names}}
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"macro_f1": macro, "weighted_f1": weighted, "per_class_f1": {names[i]: float(per[i]) for i in range(len(names))}}


def _ttf_regression_summary(part, pred_col):
    if part.empty:
        return {"mae_norm": np.nan, "mae_ticks": np.nan, "macro_run_mae_ticks": np.nan}
    abs_err_norm = np.abs(
        pd.to_numeric(part[pred_col], errors="coerce")
        - pd.to_numeric(part["y_ttf"], errors="coerce")
    )
    abs_err_ticks = abs_err_norm * TTF_NORM
    valid = np.isfinite(abs_err_ticks.to_numpy(dtype=float))
    if not valid.any():
        return {"mae_norm": np.nan, "mae_ticks": np.nan, "macro_run_mae_ticks": np.nan}
    scored = part.loc[valid, ["run_id"]].copy()
    scored["abs_err_ticks"] = abs_err_ticks.loc[valid].to_numpy()
    return {
        "mae_norm": float(scored["abs_err_ticks"].mean() / TTF_NORM),
        "mae_ticks": float(scored["abs_err_ticks"].mean()),
        "macro_run_mae_ticks": float(scored.groupby("run_id")["abs_err_ticks"].mean().mean()),
    }


def summarize(pred_df, threshold, split_name, model_name):
    y_true = pred_df["y_fail"].astype(int).to_numpy()
    probs  = pred_df["prob_fail"].to_numpy()
    bm = _binary_metrics(y_true, probs, threshold)
    out = {"model": model_name, "split": split_name, "threshold": threshold, **bm}

    pos_all = pred_df[pred_df["y_fail"].eq(1)]
    pos_near = pred_df[pred_df["has_cause"].eq(1)]
    for label, part in [("all", pos_all), ("near", pos_near)]:
        if not part.empty:
            flat = _class_report_dict(part["cause_idx"].astype(int), part["cause_pred"].astype(int), list(range(len(cause_names))), cause_names)
            hier = _class_report_dict(part["cause_idx"].astype(int), part["hier_cause_pred"].astype(int), list(range(len(cause_names))), cause_names)
            out[f"flat_cause_macro_f1_{label}"] = flat["macro_f1"]
            out[f"flat_cause_weighted_f1_{label}"] = flat["weighted_f1"]
            out[f"flat_cause_per_class_f1_{label}"] = flat["per_class_f1"]
            out[f"hier_cause_macro_f1_{label}"] = hier["macro_f1"]
            out[f"hier_cause_weighted_f1_{label}"] = hier["weighted_f1"]
            out[f"hier_cause_per_class_f1_{label}"] = hier["per_class_f1"]
        else:
            for prefix in ["flat_cause", "hier_cause"]:
                out[f"{prefix}_macro_f1_{label}"] = np.nan
                out[f"{prefix}_weighted_f1_{label}"] = np.nan
                out[f"{prefix}_per_class_f1_{label}"] = {}

    # Backward-compatible aliases.
    out["cause_macro_f1"] = out["flat_cause_macro_f1_near"]
    out["cause_weighted_f1"] = out["flat_cause_weighted_f1_near"]
    out["cause_per_class_f1"] = out["flat_cause_per_class_f1_near"]

    # Family and effusion subtype.
    fam_part = pred_df[pred_df["has_family"].eq(1)]
    fam = _class_report_dict(fam_part["family_idx"].astype(int), fam_part["family_pred"].astype(int), list(range(len(family_names))), family_names) if not fam_part.empty else {"macro_f1": np.nan, "weighted_f1": np.nan, "per_class_f1": {}}
    out["family_macro_f1_near"] = fam["macro_f1"]
    out["family_weighted_f1_near"] = fam["weighted_f1"]
    out["family_per_class_f1_near"] = fam["per_class_f1"]

    sub_part = pred_df[pred_df["has_effusion_subtype"].eq(1)]
    sub = _class_report_dict(sub_part["effusion_subtype_idx"].astype(int), sub_part["effusion_subtype_pred"].astype(int), list(range(len(effusion_subtype_names))), effusion_subtype_names) if not sub_part.empty else {"macro_f1": np.nan, "weighted_f1": np.nan, "per_class_f1": {}}
    out["effusion_subtype_macro_f1_near"] = sub["macro_f1"]
    out["effusion_subtype_weighted_f1_near"] = sub["weighted_f1"]
    out["effusion_subtype_per_class_f1_near"] = sub["per_class_f1"]

    ttf_bin = _class_report_dict(pred_df["ttf_bin_idx"].astype(int), pred_df["ttf_bin_pred"].astype(int), list(range(len(TTF_BIN_NAMES))), TTF_BIN_NAMES)
    out["ttf_bin_accuracy"] = float(accuracy_score(pred_df["ttf_bin_idx"].astype(int), pred_df["ttf_bin_pred"].astype(int)))
    out["ttf_bin_macro_f1"] = ttf_bin["macro_f1"]
    out["ttf_bin_weighted_f1"] = ttf_bin["weighted_f1"]
    out["ttf_bin_per_class_f1"] = ttf_bin["per_class_f1"]

    ttf_pos = pred_df[pred_df["has_ttf"].eq(1)]
    exact = pred_df[pred_df["has_exact_ttf"].eq(1)]

    all_blended = _ttf_regression_summary(ttf_pos, "ttf_pred")
    all_global = _ttf_regression_summary(ttf_pos, "ttf_global_pred")
    near_blended = _ttf_regression_summary(exact, "ttf_pred")
    near_specialist = _ttf_regression_summary(exact, "ttf_near_pred")

    out["shared_ttf_mae_norm"] = all_blended["mae_norm"]
    out["shared_ttf_mae_ticks"] = all_blended["mae_ticks"]
    out["shared_ttf_macro_run_mae_ticks"] = all_blended["macro_run_mae_ticks"]
    out["global_ttf_mae_ticks"] = all_global["mae_ticks"]
    out["global_ttf_macro_run_mae_ticks"] = all_global["macro_run_mae_ticks"]
    out["shared_ttf_mae_norm_near"] = near_blended["mae_norm"]
    out["shared_ttf_mae_ticks_near"] = near_blended["mae_ticks"]
    out["shared_ttf_macro_run_mae_ticks_near"] = near_blended["macro_run_mae_ticks"]
    out["near_specialist_ttf_mae_ticks"] = near_specialist["mae_ticks"]

    # Region diagnostics reveal whether a gain is broad or merely concentrated
    # in the densely sampled final countdown.
    for lo, hi, label in [
        (0.0, 30.0, "0_30"), (30.0, 60.0, "30_60"),
        (60.0, 120.0, "60_120"), (120.0, 240.0, "120_240"),
        (240.0, float("inf"), "gt_240"),
    ]:
        region = ttf_pos[
            pd.to_numeric(ttf_pos["ttf_ticks"], errors="coerce").ge(lo)
            & pd.to_numeric(ttf_pos["ttf_ticks"], errors="coerce").lt(hi)
        ]
        rm = _ttf_regression_summary(region, "ttf_pred")
        out[f"ttf_mae_ticks_{label}"] = rm["mae_ticks"]
        out[f"ttf_macro_run_mae_ticks_{label}"] = rm["macro_run_mae_ticks"]

    if not exact.empty:
        out["ttf_near_gate_mean"] = float(pd.to_numeric(exact["ttf_near_gate"], errors="coerce").mean())
        ok_true = exact["cause_ttf_pred_true_cause"].notna()
        ok_pred = exact["cause_ttf_pred_pred_cause"].notna()
        out["cause_specific_ttf_mae_ticks_true_cause"] = float(np.mean(np.abs(exact.loc[ok_true, "cause_ttf_pred_true_cause"] - exact.loc[ok_true, "y_ttf"])) * TTF_NORM) if ok_true.any() else np.nan
        out["cause_specific_ttf_mae_ticks_pred_cause"] = float(np.mean(np.abs(exact.loc[ok_pred, "cause_ttf_pred_pred_cause"] - exact.loc[ok_pred, "y_ttf"])) * TTF_NORM) if ok_pred.any() else np.nan
    else:
        out["ttf_near_gate_mean"] = np.nan
        out["cause_specific_ttf_mae_ticks_true_cause"] = np.nan
        out["cause_specific_ttf_mae_ticks_pred_cause"] = np.nan

    # Backward-compatible aliases.
    out["ttf_mae_norm"] = out["shared_ttf_mae_norm"]
    out["ttf_mae_ticks"] = out["shared_ttf_mae_ticks"]
    out["ttf_mae_norm_near"] = out["shared_ttf_mae_norm_near"]
    out["ttf_mae_ticks_near"] = out["shared_ttf_mae_ticks_near"]

    horizon_metrics = {}
    horizon_f1s = []
    for h in HORIZON_TICKS:
        tcol = f"horizon_{int(h)}_true"
        pcol = f"horizon_{int(h)}_prob"
        hm = _binary_metrics(pred_df[tcol].astype(int).to_numpy(), pred_df[pcol].to_numpy(), 0.5)
        horizon_metrics[f"within_{int(h)}"] = hm
        if np.isfinite(hm["f1"]):
            horizon_f1s.append(hm["f1"])
    out["horizon_metrics"] = horizon_metrics
    out["horizon_macro_f1"] = float(np.mean(horizon_f1s)) if horizon_f1s else np.nan

    hpos = pred_df[pred_df["has_ttf"].eq(1)]
    if not hpos.empty:
        out["horizon_estimated_ttf_mae_ticks"] = float(np.mean(np.abs(hpos["horizon_estimated_ttf_ticks"] - hpos["ttf_ticks"])))
    else:
        out["horizon_estimated_ttf_mae_ticks"] = np.nan
    hnear = pred_df[pred_df["has_exact_ttf"].eq(1)]
    if not hnear.empty:
        out["horizon_estimated_ttf_mae_ticks_near"] = float(np.mean(np.abs(hnear["horizon_estimated_ttf_ticks"] - hnear["ttf_ticks"])))
    else:
        out["horizon_estimated_ttf_mae_ticks_near"] = np.nan

    return out


def validation_score(summary):
    """Composite model-selection score for health monitoring."""
    auroc = summary.get("auroc", 0.0)
    if not np.isfinite(auroc):
        auroc = 0.0
    cause = summary.get("flat_cause_macro_f1_near", 0.0)
    if not np.isfinite(cause):
        cause = 0.0
    horizon = summary.get("horizon_macro_f1", 0.0)
    if not np.isfinite(horizon):
        horizon = 0.0
    ttf_bin = summary.get("ttf_bin_macro_f1", 0.0)
    if not np.isfinite(ttf_bin):
        ttf_bin = 0.0
    overall_ttf_mae = summary.get(
        "shared_ttf_macro_run_mae_ticks",
        summary.get("shared_ttf_mae_ticks", np.nan),
    )
    near_ttf_mae = summary.get(
        "shared_ttf_macro_run_mae_ticks_near",
        summary.get("shared_ttf_mae_ticks_near", np.nan),
    )
    overall_ttf_penalty = 0.0 if not np.isfinite(overall_ttf_mae) else float(overall_ttf_mae)
    near_ttf_penalty = 0.0 if not np.isfinite(near_ttf_mae) else float(near_ttf_mae)
    return (
        CFG["score_binary_auroc_weight"] * float(auroc)
        + CFG["score_cause_f1_weight"] * float(cause)
        + CFG["score_horizon_f1_weight"] * float(horizon)
        + CFG["score_ttf_bin_f1_weight"] * float(ttf_bin)
        - CFG["score_ttf_overall_mae_weight"] * overall_ttf_penalty
        - CFG["score_ttf_near_mae_weight"] * near_ttf_penalty
    )


def train_model(spec: ModelSpec, model_factory=None, prebuilt_ds=None,
                seed=None, save_prefix=None):
    """Train one model variant and save the composite-best checkpoint."""
    _seed = seed if seed is not None else SEED
    seed_everything(_seed)
    _tag  = save_prefix if save_prefix else spec.name[:1].upper()
    _label = save_prefix if save_prefix else spec.name
    log(f"\n{'='*60}\ntraining {_label} (seed={_seed})\n{'='*60}")

    if prebuilt_ds is not None:
        datasets, _ = prebuilt_ds
        loaders = {}
        for split in ["train", "val", "test"]:
            if split == "train":
                _wi = window_index_df[window_index_df["split"].eq("train")].reset_index(drop=True)
                _causes = _wi["cause_idx"].astype(int).values
                _near = _wi["has_cause"].astype(int).values == 1
                _sw = np.ones(len(_wi), dtype=np.float64)
                for i, c in enumerate(_causes):
                    if _wi.loc[i, "y_fail"] == 1:
                        _sw[i] = 1.5 * max(float(_wi.loc[i, "ttf_reg_weight"]), 0.5)
                    if _near[i]:
                        _sw[i] *= min(max(float(CAUSE_CLASS_WEIGHTS[c]), 1.0), 3.0) ** 0.5
                _sampler = WeightedRandomSampler(torch.from_numpy(_sw), len(_sw), replacement=True)
                loaders[split] = DataLoader(datasets[split], batch_size=CFG["batch_size"],
                                            sampler=_sampler, num_workers=0,
                                            pin_memory=torch.cuda.is_available())
            else:
                loaders[split] = DataLoader(datasets[split], batch_size=CFG["batch_size"],
                                            shuffle=False, num_workers=0,
                                            pin_memory=torch.cuda.is_available())
    else:
        datasets, loaders = make_loaders(spec, CFG["batch_size"])

    train_labels = window_index_df[window_index_df["split"].eq("train")]["y_fail"].astype(int)
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    pw_val    = negatives / max(positives, 1)
    pos_weight = torch.tensor(pw_val, dtype=torch.float32, device=DEVICE)
    log(f"{_label}: train positives={positives} negatives={negatives} pos_weight={pw_val:.3f}")

    if model_factory is not None:
        model = model_factory()
    else:
        model = SpaceForgeGraphWaveNet(
            num_nodes=len(PHYSICAL_NODES), in_dim=spec.max_node_features,
            num_causes=len(cause_names),
            fixed_supports=[s.to(DEVICE) for s in fixed_supports],
            context_dim=len(spec.context_cols),
            num_horizons=len(HORIZON_TICKS),
            num_ttf_bins=len(TTF_BIN_NAMES),
            num_families=len(family_names),
            num_effusion_subtypes=len(effusion_subtype_names),
            num_cause_ttf_heads=len(CAUSE_TTF_HEAD_NAMES),
            residual_channels=CFG["residual_channels"],
            dilation_channels=CFG["dilation_channels"],
            skip_channels=CFG["skip_channels"],
            end_channels=CFG["end_channels"],
            kernel_size=CFG["kernel_size"],
            blocks=CFG["blocks"],
            layers=CFG["layers"],
            dropout=CFG["dropout"],
            graph_order=CFG["graph_order"],
            adaptive_adj=CFG["adaptive_adj"],
            node_embedding_dim=CFG["node_embedding_dim"],
            context_dropout=CFG["context_dropout"],
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    scaler    = make_scaler()

    best_state, best_val_score, patience_left, history = None, -float("inf"), CFG["patience"], []

    for epoch in range(1, CFG["max_epochs"] + 1):
        t0 = time.perf_counter()
        train_stats = train_one_epoch(model, loaders["train"], optimizer, pos_weight, scaler, _label, epoch)
        val_pred = predict_loader(model, loaders["val"])
        val_summary = summarize(val_pred, 0.50, "val", _label)
        eps = 1e-7
        yv = val_pred["y_fail"].to_numpy(dtype=float)
        pv = np.clip(val_pred["prob_fail"].to_numpy(dtype=float), eps, 1 - eps)
        val_loss_proxy = float(-(yv * np.log(pv) + (1-yv) * np.log(1-pv)).mean())
        val_score = validation_score(val_summary)
        scheduler.step(val_score if np.isfinite(val_score) else 0.0)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_bce": train_stats["bce"],
            "train_cause": train_stats["cause"],
            "train_family": train_stats["family"],
            "train_effusion_subtype": train_stats["effusion_subtype"],
            "train_ttf_bin": train_stats["ttf_bin"],
            "train_ttf": train_stats["ttf"],
            "train_ttf_global": train_stats["ttf_global"],
            "train_ttf_near": train_stats["ttf_near"],
            "train_cause_ttf": train_stats["cause_ttf"],
            "train_horizon": train_stats["horizon"],
            "train_monotonic": train_stats["monotonic"],
            "val_loss_proxy": val_loss_proxy,
            "val_score": val_score,
            "val_f1": val_summary["f1"],
            "val_precision": val_summary["precision"],
            "val_recall": val_summary["recall"],
            "val_auroc": val_summary.get("auroc", np.nan),
            "val_cause_macro_f1_near": val_summary.get("flat_cause_macro_f1_near", np.nan),
            "val_hier_cause_macro_f1_near": val_summary.get("hier_cause_macro_f1_near", np.nan),
            "val_family_macro_f1_near": val_summary.get("family_macro_f1_near", np.nan),
            "val_ttf_bin_macro_f1": val_summary.get("ttf_bin_macro_f1", np.nan),
            "val_horizon_macro_f1": val_summary.get("horizon_macro_f1", np.nan),
            "val_ttf_mae_ticks": val_summary.get("shared_ttf_mae_ticks", np.nan),
            "val_ttf_macro_run_mae_ticks": val_summary.get("shared_ttf_macro_run_mae_ticks", np.nan),
            "val_ttf_mae_ticks_near": val_summary.get("shared_ttf_mae_ticks_near", np.nan),
            "val_ttf_macro_run_mae_ticks_near": val_summary.get("shared_ttf_macro_run_mae_ticks_near", np.nan),
            "learning_rate": lr,
            "patience_left": patience_left,
        }
        history.append(row)
        log(
            f"{_label} epoch {epoch:03d} | train={row['train_loss']:.4f} "
            f"val_score={val_score:.4f} val_auroc={row['val_auroc']:.4f} "
            f"cause_near={row['val_cause_macro_f1_near']:.4f} "
            f"ttf_bin={row['val_ttf_bin_macro_f1']:.4f} "
            f"horizon={row['val_horizon_macro_f1']:.4f} "
            f"ttf_all={row['val_ttf_mae_ticks']:.2f} "
            f"ttf_near={row['val_ttf_mae_ticks_near']:.2f} "
            f"lr={lr:.2e} patience={patience_left} elapsed={time.perf_counter()-t0:.1f}s"
        )
        if np.isfinite(val_score) and val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = CFG["patience"]
            log(f"{_label} epoch {epoch:03d} new best val_score={best_val_score:.4f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"{_label} early stopping at epoch {epoch}")
                break
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_ROOT / f"training_history_{_tag}.csv", index=False)
    torch.save(model.state_dict(), OUTPUT_ROOT / f"model_{_tag}.pt")

    _val_pred  = predict_loader(model, loaders["val"])
    _test_pred = predict_loader(model, loaders["test"])
    _val_pred.to_csv(OUTPUT_ROOT / f"predictions_val_{_tag}.csv", index=False)
    _test_pred.to_csv(OUTPUT_ROOT / f"predictions_test_{_tag}.csv", index=False)
    _val_m  = summarize(_val_pred,  0.50, "val",  _label)
    _test_m = summarize(_test_pred, 0.50, "test", _label)

    return {"model": model, "datasets": datasets, "loaders": loaders,
            "history": hist_df, "best_threshold": 0.50,
            "best_val_score": best_val_score,
            "val_metrics": _val_m, "test_metrics": _test_m, "seed": _seed}

# =============================================================================
# Train all models (multi-seed + optional baselines)
# =============================================================================
seeds         = CFG.get("seeds", [CFG["seed"]])
run_baselines = CFG.get("run_baselines", False)

log("pre-building datasets (shared across all seeds) …")
_ds_A, _ = make_loaders(MODEL_A_SPEC, CFG["batch_size"])
_ds_B, _ = make_loaders(MODEL_B_SPEC, CFG["batch_size"])

_model_keys   = (["mlp", "lstm", "A", "B"] if run_baselines else ["A", "B"])
all_seed_results = {k: [] for k in _model_keys}

for _si, _seed in enumerate(seeds):
    log(f"\n{'='*70}\nSEED {_seed}  ({_si+1}/{len(seeds)})\n{'='*70}")

    if run_baselines:
        all_seed_results["mlp"].append(train_model(
            MODEL_A_SPEC,
            model_factory=lambda: FlatMLPModel(
                num_nodes=len(PHYSICAL_NODES), in_dim=MODEL_A_SPEC.max_node_features,
                window_len=WINDOW_LEN, num_causes=len(cause_names),
                num_horizons=len(HORIZON_TICKS), num_ttf_bins=len(TTF_BIN_NAMES),
                num_families=len(family_names), num_effusion_subtypes=len(effusion_subtype_names),
                num_cause_ttf_heads=len(CAUSE_TTF_HEAD_NAMES)).to(DEVICE),
            prebuilt_ds=(_ds_A, None), seed=_seed, save_prefix=f"mlp_s{_seed}"))

        all_seed_results["lstm"].append(train_model(
            MODEL_A_SPEC,
            model_factory=lambda: NodeLSTMModel(
                num_nodes=len(PHYSICAL_NODES), in_dim=MODEL_A_SPEC.max_node_features,
                num_causes=len(cause_names), num_horizons=len(HORIZON_TICKS),
                num_ttf_bins=len(TTF_BIN_NAMES), num_families=len(family_names),
                num_effusion_subtypes=len(effusion_subtype_names),
                num_cause_ttf_heads=len(CAUSE_TTF_HEAD_NAMES)).to(DEVICE),
            prebuilt_ds=(_ds_A, None), seed=_seed, save_prefix=f"lstm_s{_seed}"))

    all_seed_results["A"].append(train_model(
        MODEL_A_SPEC, prebuilt_ds=(_ds_A, None), seed=_seed, save_prefix=f"A_s{_seed}"))

    all_seed_results["B"].append(train_model(
        MODEL_B_SPEC, prebuilt_ds=(_ds_B, None), seed=_seed, save_prefix=f"B_s{_seed}"))

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Primary results: first seed (seed 42) → used for XAI / faithfulness
results = {k: all_seed_results[k][0] for k in _model_keys}

# Multi-seed aggregation helper
def _agg(key, metric):
    vals = [r["test_metrics"].get(metric) for r in all_seed_results[key]
            if r["test_metrics"].get(metric) is not None]
    if not vals: return None, None
    return float(np.mean(vals)), float(np.std(vals))

multi_seed_summary = {}
for _k in _model_keys:
    multi_seed_summary[_k] = {
        m: {"mean": mn, "std": sd}
        for m in ["auroc", "auprc", "f1", "precision", "recall",
                  "cause_macro_f1", "cause_weighted_f1", "flat_cause_macro_f1_near", "hier_cause_macro_f1_near",
                  "family_macro_f1_near", "ttf_bin_macro_f1", "horizon_macro_f1",
                  "shared_ttf_mae_ticks", "shared_ttf_macro_run_mae_ticks",
                  "shared_ttf_mae_ticks_near", "shared_ttf_macro_run_mae_ticks_near",
                  "global_ttf_mae_ticks", "near_specialist_ttf_mae_ticks",
                  "cause_specific_ttf_mae_ticks_true_cause", "cause_specific_ttf_mae_ticks_pred_cause",
                  "horizon_estimated_ttf_mae_ticks_near"]
        for mn, sd in [_agg(_k, m)]
        if mn is not None
    }

# =============================================================================
# Compare (primary seed, all model variants)
# =============================================================================
log("running model comparison …")
comparison_rows, predictions = [], {}

_compare_pairs = [("A", MODEL_A_SPEC, MODEL_A_SPEC.name),
                  ("B", MODEL_B_SPEC, MODEL_B_SPEC.name)]
if run_baselines:
    _compare_pairs = [("mlp",  MODEL_A_SPEC, "flat_mlp"),
                      ("lstm", MODEL_A_SPEC, "node_lstm")] + _compare_pairs

for key, spec, mname in _compare_pairs:
    model     = results[key]["model"]
    threshold = results[key]["best_threshold"]
    for split in ["val","test"]:
        pred_df = predict_loader(model, results[key]["loaders"][split])
        pred_df["model_key"] = key; pred_df["split"] = split
        predictions[(key, split)] = pred_df
        s = summarize(pred_df, threshold, split, mname)
        s["model_key"] = key
        comparison_rows.append(s)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
log("model comparison:")
log(comparison_df.to_string())

# =============================================================================
# v11 XAI: target-specific masks, prototypes, stability, faithfulness, baselines
# =============================================================================
log("running v11 target-specific GNNExplainer / per-cause explanation prototypes ...")


EXPECTED_PATHS = {
    "battery_power": {
        ("array_gimbal", "solar_array"), ("solar_array", "power_bus"),
        ("battery", "battery_thermal"), ("battery_thermal", "battery"),
        ("battery", "power_bus"), ("power_bus", "heater_bank"),
    },
    "effusion_underflux": {
        ("power_bus", "heater_bank"), ("heater_bank", "effusion_cell"),
        ("source_inventory", "effusion_cell"), ("effusion_cell", "substrate"),
    },
    "effusion_undertemp": {
        ("power_bus", "heater_bank"), ("heater_bank", "effusion_cell"),
        ("source_inventory", "effusion_cell"),
    },
    "substrate_undertemp": {
        ("power_bus", "heater_bank"), ("heater_bank", "substrate"),
        ("cryo_panel", "radiator"), ("battery_thermal", "radiator"),
    },
    "stall": set(),
}


def get_item(dataset, local_idx):
    item = dataset[local_idx]
    return {k: v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()}


def select_explain_indices_by_cause(key, max_per_cause=10):
    """Prefer true positive, correct cause, high confidence, near-failure samples.

    If a cause lacks enough clean samples, this falls back to the best available
    true near-failure positives and logs the fallback.
    """
    pred = predictions[(key, "test")].copy()
    pred = pred[pred["has_cause"].eq(1)].copy()
    if pred.empty:
        log(f"Model {key}: no near-failure samples for XAI")
        return {}
    pred["correct_cause"] = pred["cause_idx"].astype(int).eq(pred["cause_pred"].astype(int))
    pred["confidence_rank"] = pred["prob_fail"].astype(float)
    out = {}
    for ci, cname in idx_to_cause.items():
        if cname == "no_failure":
            continue
        part = pred[pred["cause_idx"].eq(ci)].copy()
        if part.empty:
            log(f"Model {key}: no XAI samples for cause={cname}")
            continue
        clean = part[(part["y_fail"].eq(1)) & (part["correct_cause"])].copy()
        if len(clean) < max_per_cause:
            log(f"Model {key}: cause={cname} has only {len(clean)} clean samples; using best available {min(max_per_cause, len(part))}")
            use = part
        else:
            use = clean
        use = use.sort_values(["correct_cause", "confidence_rank"], ascending=[False, False])
        out[cname] = [int(x) for x in use.head(max_per_cause)["window_id"].tolist()]
    return out


def target_loss_for_explanation(masked, original, batch, target_kind):
    """Target-specific explainer objective.

    The objective keeps the selected target prediction stable while the mask
    shrinks, matching the GNNExplainer idea of preserving the prediction under a
    compact masked computation graph.
    """
    if target_kind == "binary":
        tgt = (torch.sigmoid(original["binary_logit"]).detach() >= 0.5).float()
        return F.binary_cross_entropy_with_logits(masked["binary_logit"], tgt)

    if target_kind == "flat_cause":
        tgt = original["cause_logits"].argmax(dim=1).detach()
        return F.cross_entropy(masked["cause_logits"], tgt)

    if target_kind == "family":
        tgt = original["family_logits"].argmax(dim=1).detach()
        return F.cross_entropy(masked["family_logits"], tgt)

    if target_kind == "horizon":
        return F.mse_loss(torch.sigmoid(masked["horizon_logits"]), torch.sigmoid(original["horizon_logits"]).detach())

    if target_kind == "ttf_bin":
        tgt = original["ttf_bin_logits"].argmax(dim=1).detach()
        return F.cross_entropy(masked["ttf_bin_logits"], tgt)

    if target_kind == "ttf_exact":
        return F.mse_loss(masked["ttf_pred"], original["ttf_pred"].detach())

    return F.binary_cross_entropy_with_logits(masked["binary_logit"], (torch.sigmoid(original["binary_logit"]).detach() >= 0.5).float())


def temporal_smoothness(mask):
    if mask.numel() <= 1:
        return mask.sum() * 0.0
    return torch.mean(torch.abs(mask[1:] - mask[:-1]))


def explain_window_target(model, dataset, local_idx, context_dim, target_kind="binary", mask_seed=0):
    """Learn edge, feature, context, and temporal masks for one target."""
    seed_everything(mask_seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    batch = get_item(dataset, local_idx)
    x, context = batch["x"], batch["context"]
    with torch.no_grad():
        original = model(x, context)
        orig_prob = torch.sigmoid(original["binary_logit"]).detach()

    n_sup = len(model.base_supports())
    edge_logits = nn.Parameter(torch.randn(n_sup, model.num_nodes, model.num_nodes, device=DEVICE) * 0.01)
    feat_logits = nn.Parameter(torch.randn(model.num_nodes, model.in_dim, device=DEVICE) * 0.01)
    temp_logits = nn.Parameter(torch.randn(x.shape[-1], device=DEVICE) * 0.01)
    params = [edge_logits, feat_logits, temp_logits]
    if context_dim > 0:
        ctx_logits = nn.Parameter(torch.randn(context_dim, device=DEVICE) * 0.01)
        params.append(ctx_logits)
    else:
        ctx_logits = None

    opt  = torch.optim.Adam(params, lr=CFG["explain_lr"])
    hist = []
    for epoch in range(1, CFG["explain_epochs"] + 1):
        opt.zero_grad(set_to_none=True)
        em = torch.sigmoid(edge_logits)
        fm = torch.sigmoid(feat_logits)
        tm = torch.sigmoid(temp_logits)
        cm = torch.sigmoid(ctx_logits) if ctx_logits is not None else None
        masked = model(x, context, support_masks=[em[i] for i in range(n_sup)],
                       feature_mask=fm, context_mask=cm, temporal_mask=tm)
        pred_loss = target_loss_for_explanation(masked, original, batch, target_kind)
        loss = (
            pred_loss
            + CFG["edge_size_penalty"]       * em.mean()
            + CFG["edge_entropy_penalty"]    * mask_entropy(em).mean()
            + CFG["feature_size_penalty"]    * fm.mean()
            + CFG["feature_entropy_penalty"] * mask_entropy(fm).mean()
            + CFG["temporal_size_penalty"]   * tm.mean()
            + CFG["temporal_entropy_penalty"]* mask_entropy(tm).mean()
            + CFG["temporal_smoothness_penalty"] * temporal_smoothness(tm)
        )
        if cm is not None:
            loss = loss + CFG["context_size_penalty"] * cm.mean() + CFG["context_entropy_penalty"] * mask_entropy(cm).mean()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            hist.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "pred_loss": float(pred_loss.detach().cpu()),
                "masked_prob": float(torch.sigmoid(masked["binary_logit"]).detach().cpu().item()),
                "original_prob": float(orig_prob.cpu().item()),
            })

    result = {
        "local_idx": local_idx,
        "window_id": int(batch["window_id"].detach().cpu().item()),
        "target_kind": target_kind,
        "mask_seed": mask_seed,
        "original_prob": float(orig_prob.cpu().item()),
        "target_binary": int((orig_prob.cpu().item() >= 0.5)),
        "target_cause_idx": int(original["cause_logits"].argmax(dim=1).detach().cpu().item()),
        "target_family_idx": int(original["family_logits"].argmax(dim=1).detach().cpu().item()),
        "target_ttf_bin_idx": int(original["ttf_bin_logits"].argmax(dim=1).detach().cpu().item()),
        "edge_masks":    torch.sigmoid(edge_logits).detach().cpu().numpy(),
        "feature_mask":  torch.sigmoid(feat_logits).detach().cpu().numpy(),
        "temporal_mask": torch.sigmoid(temp_logits).detach().cpu().numpy(),
        "context_mask":  torch.sigmoid(ctx_logits).detach().cpu().numpy() if ctx_logits is not None else None,
        "history":       pd.DataFrame(hist),
    }
    for p in model.parameters():
        p.requires_grad_(True)
    return result


def support_name(i):
    return ["forward_physical", "backward_physical", "adaptive"][i] if i < 3 else f"support_{i}"


def rank_edges(edge_masks, top_k=20):
    rows = []
    physical_edge_set = set(PHYSICAL_EDGES)
    for s in range(edge_masks.shape[0]):
        for si in range(edge_masks.shape[1]):
            for di in range(edge_masks.shape[2]):
                if si == di:
                    continue
                src, dst = idx_to_node[si], idx_to_node[di]
                is_phys = (src, dst) in physical_edge_set
                rows.append({
                    "support": support_name(s),
                    "src": src,
                    "dst": dst,
                    "edge": f"{src}->{dst}",
                    "importance": float(edge_masks[s, si, di]),
                    "is_physical_edge": bool(is_phys),
                    "is_adaptive_nonphysical": bool(s == 2 and not is_phys),
                })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def rank_node_features(feature_mask, spec, top_k=30):
    rows = []
    for ni, node in enumerate(PHYSICAL_NODES):
        for fi in range(feature_mask.shape[1]):
            name = spec.node_feature_names_by_dim[node][fi]
            if name.startswith("__pad_"):
                continue
            rows.append({"node": node, "feature": name, "importance": float(feature_mask[ni, fi])})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def rank_context(context_mask, spec, top_k=30):
    if context_mask is None or not spec.context_cols:
        return pd.DataFrame(columns=["feature", "importance"])
    rows = [{"feature": col, "importance": float(context_mask[i])} for i, col in enumerate(spec.context_cols)]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def temporal_segment_scores(temporal_mask):
    rows = []
    T = len(temporal_mask)
    for seg in CFG.get("xai_temporal_segments", [5, 10, 20, 30, 60]):
        k = min(int(seg), T)
        rows.append({"segment": f"last_{k}_ticks", "mean_importance": float(np.mean(temporal_mask[-k:])), "sum_importance": float(np.sum(temporal_mask[-k:]))})
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)


def expected_path_hit_rate(edge_df, cause_name, top_k=5):
    expected = EXPECTED_PATHS.get(cause_name, set())
    if not expected:
        return np.nan
    top = edge_df.head(top_k)
    hits = sum((r["src"], r["dst"]) in expected for _, r in top.iterrows())
    return float(hits / max(len(top), 1))


def context_reliance(exp):
    if exp["context_mask"] is None:
        return 0.0
    s = float(np.mean(exp["feature_mask"]))
    c = float(np.mean(exp["context_mask"]))
    return c / (s + c) if (s + c) > 1e-8 else 0.0


def save_heatmap(matrix, path, title="", xticks=None, yticks=None):
    if plt is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix, aspect="auto")
        ax.set_title(title)
        if xticks is not None and len(xticks) == matrix.shape[1]:
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, rotation=90, fontsize=7)
        if yticks is not None and len(yticks) == matrix.shape[0]:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks, fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        log(f"WARNING: heatmap save failed for {path}: {exc}")


@torch.no_grad()
def compute_train_means(dataset, n_sample=2000):
    n = min(n_sample, len(dataset))
    idxs = np.random.default_rng(0).choice(len(dataset), n, replace=False)
    x_sum = None
    ctx_sum = None
    for i in idxs:
        item = dataset[i]
        xm   = item["x"].float().mean(dim=-1)
        ctxm = item["context"].float().mean(dim=-1)
        if x_sum is None:
            x_sum = xm
            ctx_sum = ctxm
        else:
            x_sum = x_sum + xm
            ctx_sum = ctx_sum + ctxm
    return (x_sum / n).to(DEVICE), (ctx_sum / n).to(DEVICE)


@torch.no_grad()
def score_with_masks(model, dataset, local_idx, feature_mean=None, context_mean=None,
                     feature_binary_mask=None, context_binary_mask=None,
                     temporal_binary_mask=None, support_masks=None):
    model.eval()
    batch = get_item(dataset, local_idx)
    x   = batch["x"].clone()
    ctx = batch["context"].clone()
    if feature_binary_mask is not None and feature_mean is not None:
        fm = feature_binary_mask.to(DEVICE).view(1, feature_binary_mask.shape[0], feature_binary_mask.shape[1], 1)
        x = x * fm + (1.0 - fm) * feature_mean.view(1, feature_binary_mask.shape[0], feature_binary_mask.shape[1], 1)
    if context_binary_mask is not None and context_mean is not None and ctx.numel() > 0:
        cm = context_binary_mask.to(DEVICE).view(1, -1, 1)
        ctx = ctx * cm + (1.0 - cm) * context_mean.view(1, -1, 1)
    if temporal_binary_mask is not None:
        tm = temporal_binary_mask.to(DEVICE)
        tm_x = tm.view(1, 1, 1, -1)
        x = x * tm_x + (1.0 - tm_x) * x.mean(dim=-1, keepdim=True)
        if ctx.numel() > 0:
            tm_c = tm.view(1, 1, -1)
            ctx = ctx * tm_c + (1.0 - tm_c) * ctx.mean(dim=-1, keepdim=True)
    out = model(x, ctx, support_masks=support_masks)
    return float(torch.sigmoid(out["binary_logit"]).detach().cpu().item())


def make_top_feature_mask(exp, k, keep_only=False, random_baseline=False, rng=None):
    vals = exp["feature_mask"].reshape(-1)
    valid = np.ones_like(vals, dtype=bool)
    rng = np.random.default_rng(0) if rng is None else rng
    if random_baseline:
        idxs = rng.choice(np.where(valid)[0], size=min(k, valid.sum()), replace=False)
    else:
        idxs = np.argsort(vals)[::-1][:k]
    mask = np.zeros_like(vals, dtype=np.float32) if keep_only else np.ones_like(vals, dtype=np.float32)
    mask[idxs] = 1.0 if keep_only else 0.0
    return torch.tensor(mask.reshape(exp["feature_mask"].shape), dtype=torch.float32, device=DEVICE)


def make_top_temporal_mask(exp, k, keep_only=False):
    vals = exp["temporal_mask"]
    idxs = np.argsort(vals)[::-1][:min(k, len(vals))]
    mask = np.zeros_like(vals, dtype=np.float32) if keep_only else np.ones_like(vals, dtype=np.float32)
    mask[idxs] = 1.0 if keep_only else 0.0
    return torch.tensor(mask, dtype=torch.float32, device=DEVICE)


def make_top_context_mask(exp, k, keep_only=False, random_baseline=False, rng=None):
    if exp["context_mask"] is None:
        return None
    vals = exp["context_mask"]
    rng = np.random.default_rng(0) if rng is None else rng
    if random_baseline:
        idxs = rng.choice(np.arange(len(vals)), size=min(k, len(vals)), replace=False)
    else:
        idxs = np.argsort(vals)[::-1][:min(k, len(vals))]
    mask = np.zeros_like(vals, dtype=np.float32) if keep_only else np.ones_like(vals, dtype=np.float32)
    mask[idxs] = 1.0 if keep_only else 0.0
    return torch.tensor(mask, dtype=torch.float32, device=DEVICE)


def make_top_edge_masks(exp, k, keep_only=False, random_baseline=False, rng=None):
    vals = exp["edge_masks"].reshape(-1)
    rng = np.random.default_rng(0) if rng is None else rng
    if random_baseline:
        idxs = rng.choice(np.arange(len(vals)), size=min(k, len(vals)), replace=False)
    else:
        idxs = np.argsort(vals)[::-1][:min(k, len(vals))]
    mask = np.zeros_like(vals, dtype=np.float32) if keep_only else np.ones_like(vals, dtype=np.float32)
    mask[idxs] = 1.0 if keep_only else 0.0
    mask = mask.reshape(exp["edge_masks"].shape)
    return [torch.tensor(mask[i], dtype=torch.float32, device=DEVICE) for i in range(mask.shape[0])]


def gradient_saliency(model, dataset, local_idx, target_kind):
    model.eval()
    batch = get_item(dataset, local_idx)
    x = batch["x"].clone().detach().requires_grad_(True)
    ctx = batch["context"].clone().detach()
    out = model(x, ctx)
    if target_kind == "binary":
        scalar = out["binary_logit"].sum()
    elif target_kind == "flat_cause":
        scalar = out["cause_logits"][0, out["cause_logits"].argmax(dim=1).item()]
    elif target_kind == "family":
        scalar = out["family_logits"][0, out["family_logits"].argmax(dim=1).item()]
    elif target_kind == "ttf_bin":
        scalar = out["ttf_bin_logits"][0, out["ttf_bin_logits"].argmax(dim=1).item()]
    elif target_kind == "horizon":
        scalar = out["horizon_logits"].max()
    else:
        scalar = out["ttf_pred"].sum()
    scalar.backward()
    sal = x.grad.detach().abs().mean(dim=-1).squeeze(0).cpu().numpy()
    return sal


def occlusion_importance_features(model, dataset, local_idx, feature_mean, top_k=20):
    base = score_with_masks(model, dataset, local_idx)
    item = get_item(dataset, local_idx)
    x = item["x"].clone()
    ctx = item["context"].clone()
    rows = []
    with torch.no_grad():
        for ni, node in enumerate(PHYSICAL_NODES):
            for fi in range(x.shape[2]):
                x_occ = x.clone()
                x_occ[:, ni, fi, :] = feature_mean[ni, fi]
                p = float(torch.sigmoid(model(x_occ, ctx)["binary_logit"]).cpu().item())
                rows.append({"node": node, "feature_dim": fi, "prob_delta": base - p})
    return pd.DataFrame(rows).sort_values("prob_delta", ascending=False).head(top_k).reset_index(drop=True)


# Run XAI.
explanations = {}
explanations_by_cause = {}
ranking_outputs = {}
interp_rows = []
faith_rows = []
xai_stability_rows = []
expected_path_rows = []
temporal_segment_rows = []
context_ablation_rows = []
xai_baseline_rows = []

xai_model_keys = [k for k in CFG.get("xai_model_keys", ["A", "B"]) if k in results]
target_kinds = CFG.get("xai_target_kinds", ["binary"])
mask_seeds = CFG.get("xai_mask_seeds", [11, 22, 33])
top_k_values = CFG.get("xai_top_k_values", [1, 3, 5, 10])

for key in xai_model_keys:
    spec = MODEL_A_SPEC if key == "A" else MODEL_B_SPEC
    model = results[key]["model"]
    ds_train = results[key]["datasets"]["train"]
    ds_test = results[key]["datasets"]["test"]
    feat_mean, ctx_mean = compute_train_means(ds_train)
    idx_by_cause = select_explain_indices_by_cause(key, CFG.get("xai_samples_per_cause", 10))
    explanations_by_cause[key] = {}

    # Model B context ablations. no-context uses train-set means. context-only
    # uses subsystem train means and keeps context intact.
    if key == "B":
        pred = predictions[(key, "test")]
        use_idxs = pred[pred["has_cause"].eq(1)]["window_id"].head(50).astype(int).tolist()
        for idx in use_idxs:
            base = score_with_masks(model, ds_test, idx)
            no_ctx_mask = torch.zeros(len(spec.context_cols), dtype=torch.float32, device=DEVICE)
            no_ctx = score_with_masks(model, ds_test, idx, context_mean=ctx_mean, context_binary_mask=no_ctx_mask)
            x_mean_mask = torch.zeros((len(PHYSICAL_NODES), spec.max_node_features), dtype=torch.float32, device=DEVICE)
            context_only = score_with_masks(model, ds_test, idx, feature_mean=feat_mean, feature_binary_mask=x_mean_mask)
            context_ablation_rows.append({"model_key": key, "window_id": idx, "base_prob": base, "no_context_prob": no_ctx, "context_only_prob": context_only})

    for cause_name, idxs in idx_by_cause.items():
        explanations_by_cause[key][cause_name] = {}
        for target_kind in target_kinds:
            all_seed_exps = []
            for local_idx in idxs:
                for ms in mask_seeds:
                    log(f"Model {key} XAI cause={cause_name} target={target_kind} window={local_idx} mask_seed={ms}")
                    exp = explain_window_target(model, ds_test, local_idx, len(spec.context_cols), target_kind=target_kind, mask_seed=ms)
                    exp["cause_name"] = cause_name
                    all_seed_exps.append(exp)
                    exp["history"].to_csv(OUTPUT_ROOT / f"xai_history_model_{key}_{cause_name}_{target_kind}_w{local_idx}_s{ms}.csv", index=False)

            if not all_seed_exps:
                continue

            edge_stack = np.stack([e["edge_masks"] for e in all_seed_exps])
            feat_stack = np.stack([e["feature_mask"] for e in all_seed_exps])
            temp_stack = np.stack([e["temporal_mask"] for e in all_seed_exps])
            ctx_stack = np.stack([e["context_mask"] for e in all_seed_exps]) if spec.context_cols and all(e["context_mask"] is not None for e in all_seed_exps) else None

            proto = {
                "local_idx": all_seed_exps[0]["local_idx"],
                "window_id": all_seed_exps[0]["window_id"],
                "target_kind": target_kind,
                "cause_name": cause_name,
                "original_prob": float(np.mean([e["original_prob"] for e in all_seed_exps])),
                "edge_masks": edge_stack.mean(axis=0),
                "edge_masks_std": edge_stack.std(axis=0),
                "feature_mask": feat_stack.mean(axis=0),
                "feature_mask_std": feat_stack.std(axis=0),
                "temporal_mask": temp_stack.mean(axis=0),
                "temporal_mask_std": temp_stack.std(axis=0),
                "context_mask": ctx_stack.mean(axis=0) if ctx_stack is not None else None,
                "context_mask_std": ctx_stack.std(axis=0) if ctx_stack is not None else None,
            }
            explanations_by_cause[key][cause_name][target_kind] = proto
            if key not in explanations and target_kind == "binary":
                explanations[key] = proto

            er = rank_edges(proto["edge_masks"], top_k=50)
            fr = rank_node_features(proto["feature_mask"], spec, top_k=50)
            cr = rank_context(proto["context_mask"], spec, top_k=50)
            ts = temporal_segment_scores(proto["temporal_mask"])
            er["model_key"] = key; er["cause"] = cause_name; er["target_kind"] = target_kind
            fr["model_key"] = key; fr["cause"] = cause_name; fr["target_kind"] = target_kind
            cr["model_key"] = key; cr["cause"] = cause_name; cr["target_kind"] = target_kind
            ts["model_key"] = key; ts["cause"] = cause_name; ts["target_kind"] = target_kind

            er.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_edges.csv", index=False)
            er[er["support"].str.contains("physical")].to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_physical_edges.csv", index=False)
            er[er["support"].eq("adaptive")].to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_adaptive_edges.csv", index=False)
            fr.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_features.csv", index=False)
            cr.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_context.csv", index=False)
            ts.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_temporal_segments.csv", index=False)

            # Also save the exact filenames requested, using Model B binary
            # prototypes when available; otherwise Model A.
            if target_kind == "binary" and (key == "B" or not (OUTPUT_ROOT / f"xai_{cause_name}_edges.csv").exists()):
                er.to_csv(OUTPUT_ROOT / f"xai_{cause_name}_edges.csv", index=False)
                fr.to_csv(OUTPUT_ROOT / f"xai_{cause_name}_features.csv", index=False)
                cr.to_csv(OUTPUT_ROOT / f"xai_{cause_name}_context.csv", index=False)
                ts.to_csv(OUTPUT_ROOT / f"xai_{cause_name}_temporal.csv", index=False)

            for k_top in top_k_values:
                expected_path_rows.append({
                    "model_key": key, "cause": cause_name, "target_kind": target_kind,
                    "top_k": k_top, "expected_path_hit_rate": expected_path_hit_rate(er, cause_name, k_top)
                })
            temporal_segment_rows.extend(ts.to_dict(orient="records"))

            # Stability summaries.
            xai_stability_rows.append({
                "model_key": key, "cause": cause_name, "target_kind": target_kind,
                "edge_importance_mean": float(edge_stack.mean()),
                "edge_importance_std": float(edge_stack.std()),
                "feature_importance_mean": float(feat_stack.mean()),
                "feature_importance_std": float(feat_stack.std()),
                "temporal_importance_mean": float(temp_stack.mean()),
                "temporal_importance_std": float(temp_stack.std()),
                "context_importance_mean": float(ctx_stack.mean()) if ctx_stack is not None else np.nan,
                "context_importance_std": float(ctx_stack.std()) if ctx_stack is not None else np.nan,
            })

            # PNG heatmaps.
            save_heatmap(proto["edge_masks"][0], OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_edge_forward.png", f"{key} {cause_name} {target_kind} forward", PHYSICAL_NODES, PHYSICAL_NODES)
            if proto["edge_masks"].shape[0] > 1:
                save_heatmap(proto["edge_masks"][1], OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_edge_backward.png", f"{key} {cause_name} {target_kind} backward", PHYSICAL_NODES, PHYSICAL_NODES)
            if proto["edge_masks"].shape[0] > 2:
                save_heatmap(proto["edge_masks"][2], OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_edge_adaptive.png", f"{key} {cause_name} {target_kind} adaptive", PHYSICAL_NODES, PHYSICAL_NODES)
            save_heatmap(proto["feature_mask"], OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_features.png", f"{key} {cause_name} {target_kind} features")
            save_heatmap(proto["temporal_mask"].reshape(1, -1), OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_time.png", f"{key} {cause_name} {target_kind} time")
            if proto["context_mask"] is not None:
                save_heatmap(proto["context_mask"].reshape(1, -1), OUTPUT_ROOT / f"xai_png/{key}_{cause_name}_{target_kind}_context.png", f"{key} {cause_name} {target_kind} context")

            # Baselines: gradient saliency, occlusion, adaptive adjacency weights.
            try:
                sal = gradient_saliency(model, ds_test, proto["local_idx"], target_kind)
                sal_df = pd.DataFrame([
                    {"node": PHYSICAL_NODES[ni], "feature_dim": fi, "importance": float(sal[ni, fi])}
                    for ni in range(sal.shape[0]) for fi in range(sal.shape[1])
                ]).sort_values("importance", ascending=False)
                sal_df.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_gradient_saliency.csv", index=False)
                xai_baseline_rows.append({"model_key": key, "cause": cause_name, "target_kind": target_kind, "baseline": "gradient_saliency", "top_importance": float(sal_df.iloc[0]["importance"]) if not sal_df.empty else np.nan})
            except Exception as exc:
                log(f"WARNING: gradient saliency failed for {key}/{cause_name}/{target_kind}: {exc}")
            try:
                occ_df = occlusion_importance_features(model, ds_test, proto["local_idx"], feat_mean)
                occ_df.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_occlusion_importance.csv", index=False)
                xai_baseline_rows.append({"model_key": key, "cause": cause_name, "target_kind": target_kind, "baseline": "occlusion", "top_importance": float(occ_df.iloc[0]["prob_delta"]) if not occ_df.empty else np.nan})
            except Exception as exc:
                log(f"WARNING: occlusion failed for {key}/{cause_name}/{target_kind}: {exc}")
            try:
                supports = model.base_supports()
                if len(supports) > 2:
                    adp = supports[2].detach().cpu().numpy()
                    adp_df = rank_edges(np.stack([np.zeros_like(adp), np.zeros_like(adp), adp]), top_k=50)
                    adp_df.to_csv(OUTPUT_ROOT / f"xai_{key}_{cause_name}_{target_kind}_adaptive_adjacency_weights.csv", index=False)
                    xai_baseline_rows.append({"model_key": key, "cause": cause_name, "target_kind": target_kind, "baseline": "adaptive_adjacency_weights", "top_importance": float(adp_df.iloc[0]["importance"]) if not adp_df.empty else np.nan})
            except Exception as exc:
                log(f"WARNING: adaptive adjacency baseline failed for {key}/{cause_name}/{target_kind}: {exc}")

            # Faithfulness: remove top-k, keep-only top-k, and random baselines.
            base = score_with_masks(model, ds_test, proto["local_idx"])
            rng = np.random.default_rng(123)
            for k_top in top_k_values:
                for kind, keep_only in [("remove_top", False), ("keep_only", True)]:
                    mp_feat = score_with_masks(model, ds_test, proto["local_idx"], feature_mean=feat_mean,
                                               feature_binary_mask=make_top_feature_mask(proto, k_top, keep_only=keep_only))
                    mp_edge = score_with_masks(model, ds_test, proto["local_idx"],
                                               support_masks=make_top_edge_masks(proto, k_top, keep_only=keep_only))
                    mp_time = score_with_masks(model, ds_test, proto["local_idx"],
                                               temporal_binary_mask=make_top_temporal_mask(proto, min(k_top, WINDOW_LEN), keep_only=keep_only))
                    faith_rows += [
                        {"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": kind, "drop_type": "features", "top_k": k_top, "base_prob": base, "masked_prob": mp_feat, "prob_delta": base - mp_feat},
                        {"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": kind, "drop_type": "edges", "top_k": k_top, "base_prob": base, "masked_prob": mp_edge, "prob_delta": base - mp_edge},
                        {"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": kind, "drop_type": "time", "top_k": k_top, "base_prob": base, "masked_prob": mp_time, "prob_delta": base - mp_time},
                    ]
                    cm = make_top_context_mask(proto, k_top, keep_only=keep_only)
                    if cm is not None:
                        mp_ctx = score_with_masks(model, ds_test, proto["local_idx"], context_mean=ctx_mean, context_binary_mask=cm)
                        faith_rows.append({"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": kind, "drop_type": "context", "top_k": k_top, "base_prob": base, "masked_prob": mp_ctx, "prob_delta": base - mp_ctx})

                # Random baseline for remove top.
                rnd_feat = score_with_masks(model, ds_test, proto["local_idx"], feature_mean=feat_mean,
                                            feature_binary_mask=make_top_feature_mask(proto, k_top, keep_only=False, random_baseline=True, rng=rng))
                rnd_edge = score_with_masks(model, ds_test, proto["local_idx"],
                                            support_masks=make_top_edge_masks(proto, k_top, keep_only=False, random_baseline=True, rng=rng))
                faith_rows += [
                    {"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": "remove_random", "drop_type": "features", "top_k": k_top, "base_prob": base, "masked_prob": rnd_feat, "prob_delta": base - rnd_feat},
                    {"model_key": key, "cause": cause_name, "target_kind": target_kind, "masking_baseline": "train_mean_or_empirical", "faithfulness_type": "remove_random", "drop_type": "edges", "top_k": k_top, "base_prob": base, "masked_prob": rnd_edge, "prob_delta": base - rnd_edge},
                ]

# Build top-level binary summaries for old summary structure.
for key in xai_model_keys:
    spec = MODEL_A_SPEC if key == "A" else MODEL_B_SPEC
    binary_protos = []
    for cause_dict in explanations_by_cause.get(key, {}).values():
        if "binary" in cause_dict:
            binary_protos.append(cause_dict["binary"])
    if binary_protos:
        edge_proto = np.mean(np.stack([p["edge_masks"] for p in binary_protos]), axis=0)
        feat_proto = np.mean(np.stack([p["feature_mask"] for p in binary_protos]), axis=0)
        temp_proto = np.mean(np.stack([p["temporal_mask"] for p in binary_protos]), axis=0)
        ctx_proto = np.mean(np.stack([p["context_mask"] for p in binary_protos]), axis=0) if spec.context_cols and all(p["context_mask"] is not None for p in binary_protos) else None
        explanations[key] = {"edge_masks": edge_proto, "feature_mask": feat_proto, "temporal_mask": temp_proto, "context_mask": ctx_proto, "original_prob": float(np.mean([p["original_prob"] for p in binary_protos]))}
        ranking_outputs[(key, "edges")] = rank_edges(edge_proto, 30)
        ranking_outputs[(key, "features")] = rank_node_features(feat_proto, spec, 30)
        ranking_outputs[(key, "context")] = rank_context(ctx_proto, spec, 30)
        interp_rows.append({
            "model_key": key,
            "model_name": spec.name,
            "prototype_fail_probability_mean": explanations[key]["original_prob"],
            "top_edge": f"{ranking_outputs[(key,'edges')].iloc[0]['src']} -> {ranking_outputs[(key,'edges')].iloc[0]['dst']}" if not ranking_outputs[(key,"edges")].empty else "none",
            "top_feature": f"{ranking_outputs[(key,'features')].iloc[0]['node']}::{ranking_outputs[(key,'features')].iloc[0]['feature']}" if not ranking_outputs[(key,"features")].empty else "none",
            "top_context": str(ranking_outputs[(key,'context')].iloc[0]["feature"]) if not ranking_outputs[(key,"context")].empty else "none",
            "context_reliance_score": context_reliance(explanations[key]),
        })
    else:
        ranking_outputs[(key, "edges")] = pd.DataFrame()
        ranking_outputs[(key, "features")] = pd.DataFrame()
        ranking_outputs[(key, "context")] = pd.DataFrame()

faith_df = pd.DataFrame(faith_rows)
faith_df.to_csv(OUTPUT_ROOT / "faithfulness_checks.csv", index=False)
interp_df = pd.DataFrame(interp_rows)
interp_df.to_csv(OUTPUT_ROOT / "interpretation_summary.csv", index=False)
pd.DataFrame(xai_stability_rows).to_csv(OUTPUT_ROOT / "xai_stability_summary.csv", index=False)
pd.DataFrame(expected_path_rows).to_csv(OUTPUT_ROOT / "xai_expected_path_hit_rates.csv", index=False)
pd.DataFrame(temporal_segment_rows).to_csv(OUTPUT_ROOT / "xai_temporal_segment_scores.csv", index=False)
pd.DataFrame(context_ablation_rows).to_csv(OUTPUT_ROOT / "xai_model_b_context_ablation.csv", index=False)
pd.DataFrame(xai_baseline_rows).to_csv(OUTPUT_ROOT / "xai_baseline_comparison.csv", index=False)

# Save dedicated metrics CSVs.
cause_metric_rows = []
horizon_metric_rows = []
ttf_bin_metric_rows = []
ttf_reg_rows = []
for _, row in comparison_df.iterrows():
    model_key = row["model_key"]
    split = row["split"]
    for family_prefix in ["flat_cause", "hier_cause"]:
        per = row.get(f"{family_prefix}_per_class_f1_near", {})
        if isinstance(per, dict):
            for cls, val in per.items():
                cause_metric_rows.append({"model_key": model_key, "split": split, "metric_family": family_prefix, "class_name": cls, "f1_near": val})
    hm = row.get("horizon_metrics", {})
    if isinstance(hm, dict):
        for hname, vals in hm.items():
            vals = vals if isinstance(vals, dict) else {}
            horizon_metric_rows.append({"model_key": model_key, "split": split, "horizon": hname, **vals})
    per_bin = row.get("ttf_bin_per_class_f1", {})
    if isinstance(per_bin, dict):
        for cls, val in per_bin.items():
            ttf_bin_metric_rows.append({"model_key": model_key, "split": split, "ttf_bin": cls, "f1": val})
    ttf_reg_rows.append({
        "model_key": model_key, "split": split,
        "shared_ttf_mae_ticks": row.get("shared_ttf_mae_ticks", np.nan),
        "shared_ttf_macro_run_mae_ticks": row.get("shared_ttf_macro_run_mae_ticks", np.nan),
        "shared_ttf_mae_ticks_near": row.get("shared_ttf_mae_ticks_near", np.nan),
        "shared_ttf_macro_run_mae_ticks_near": row.get("shared_ttf_macro_run_mae_ticks_near", np.nan),
        "global_ttf_mae_ticks": row.get("global_ttf_mae_ticks", np.nan),
        "near_specialist_ttf_mae_ticks": row.get("near_specialist_ttf_mae_ticks", np.nan),
        "ttf_mae_ticks_0_30": row.get("ttf_mae_ticks_0_30", np.nan),
        "ttf_mae_ticks_30_60": row.get("ttf_mae_ticks_30_60", np.nan),
        "ttf_mae_ticks_60_120": row.get("ttf_mae_ticks_60_120", np.nan),
        "ttf_mae_ticks_120_240": row.get("ttf_mae_ticks_120_240", np.nan),
        "ttf_mae_ticks_gt_240": row.get("ttf_mae_ticks_gt_240", np.nan),
        "cause_specific_ttf_mae_ticks_true_cause": row.get("cause_specific_ttf_mae_ticks_true_cause", np.nan),
        "cause_specific_ttf_mae_ticks_pred_cause": row.get("cause_specific_ttf_mae_ticks_pred_cause", np.nan),
        "horizon_estimated_ttf_mae_ticks": row.get("horizon_estimated_ttf_mae_ticks", np.nan),
        "horizon_estimated_ttf_mae_ticks_near": row.get("horizon_estimated_ttf_mae_ticks_near", np.nan),
    })
pd.DataFrame(cause_metric_rows).to_csv(OUTPUT_ROOT / "per_class_cause_metrics.csv", index=False)
pd.DataFrame(horizon_metric_rows).to_csv(OUTPUT_ROOT / "per_horizon_metrics.csv", index=False)
pd.DataFrame(ttf_bin_metric_rows).to_csv(OUTPUT_ROOT / "ttf_bin_metrics.csv", index=False)
pd.DataFrame(ttf_reg_rows).to_csv(OUTPUT_ROOT / "ttf_regression_metrics.csv", index=False)

# =============================================================================
# run_summary.json
# =============================================================================
def safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


def safe_record_dict(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out[k] = {kk: safe(vv) for kk, vv in v.items()}
        else:
            out[k] = safe(v)
    return out


summary = {
    "timestamp": datetime.now().isoformat(),
    "cfg": CFG,
    "v11_design": {
        "dense_horizons": HORIZON_TICKS,
        "ttf_bins": TTF_BIN_NAMES,
        "cause_near_window_ticks": CAUSE_NEAR_WINDOW_TICKS,
        "ttf_near_window_ticks": TTF_NEAR_WINDOW_TICKS,
        "ttf_refine_window_ticks": TTF_REFINE_WINDOW_TICKS,
        "ttf_blend_horizon": HORIZON_TICKS[TTF_BLEND_HORIZON_IDX],
        "ttf_regression_region_weights": TTF_REG_WEIGHTS.tolist(),
        "near_failure_stride": NEAR_FAILURE_STRIDE,
        "near_failure_stride_window_ticks": NEAR_FAILURE_STRIDE_WINDOW_TICKS,
        "hierarchical_cause_families": family_names,
        "effusion_subtypes": effusion_subtype_names,
        "cause_specific_ttf_heads": CAUSE_TTF_HEAD_NAMES,
        "physical_nodes": PHYSICAL_NODES,
        "expansion_nodes": EXPANSION_NODES,
    },
    "dataset": {
        "total_runs": total_runs,
        "failed_runs": failed_runs,
        "alive_runs": alive_runs,
        "failure_rate": round(failed_runs / total_runs, 4) if total_runs else None,
        "cause_distribution": run_meta["coarse_cause"].value_counts().to_dict(),
        "family_distribution": run_meta["family_name"].value_counts().to_dict(),
        "n_configs": int(run_meta["config_key"].nunique()),
    },
    "split": {
        s: {"configs": int(run_meta[run_meta["split"].eq(s)]["config_key"].nunique()),
            "runs": int(run_meta[run_meta["split"].eq(s)]["run_id"].nunique()),
            "failures": int(run_meta[run_meta["split"].eq(s)]["y_fail_run"].sum())}
        for s in ["train","val","test"]
    },
    "windows": {
        s: {
            "total": int(window_index_df[window_index_df["split"].eq(s)].shape[0]),
            "positives": int(window_index_df[window_index_df["split"].eq(s)]["y_fail"].sum()),
            "near_cause": int(window_index_df[window_index_df["split"].eq(s)]["has_cause"].sum()),
            "all_ttf": int(window_index_df[window_index_df["split"].eq(s)]["has_ttf"].sum()),
            "near_ttf": int(window_index_df[window_index_df["split"].eq(s)]["has_exact_ttf"].sum()),
            "refine_ttf": int(window_index_df[window_index_df["split"].eq(s)]["has_refine_ttf"].sum()),
            "stride1": int((window_index_df[window_index_df["split"].eq(s)]["stride_used"] == 1).sum()),
        }
        for s in ["train","val","test"]
    },
    "features": {
        "node_feature_counts": {n: len(node_cols_by_node[n]) for n in PHYSICAL_NODES},
        "context_cols_model_b": len(context_cols_b),
        "model_a_max_node_features": MODEL_A_SPEC.max_node_features,
        "model_b_max_node_features": MODEL_B_SPEC.max_node_features,
        "engineered_v11_features": [
            c for c in [
                "battery_soc_slope_5", "battery_soc_slope_20",
                "power_deficit_w", "power_deficit_rolling_sum_10", "power_deficit_rolling_sum_30",
                "effusion_temp_error", "effusion_temp_error_slope_5", "effusion_temp_error_slope_20",
                "flux_ratio", "flux_ratio_slope_5", "flux_ratio_slope_20",
                "substrate_temp_error", "substrate_temp_error_slope_5", "substrate_temp_error_slope_20",
                "battery_thermal_pack_temp_slope_5", "battery_thermal_pack_temp_slope_20",
                "battery_thermal_sink_delta_k",
                "radiator_loop_temp_slope_5", "radiator_loop_temp_slope_20",
                "radiator_heat_balance_w", "radiator_heat_balance_rolling_sum_20",
                "source_inventory_remaining_frac_slope_5", "source_inventory_remaining_frac_slope_20",
                "array_gimbal_pointing_eff_slope_5", "array_gimbal_pointing_eff_slope_20",
                "array_gimbal_pointing_err_slope_5", "array_gimbal_pointing_err_slope_20",
                "cryo_panel_adsorbed_mass_slope_5", "cryo_panel_adsorbed_mass_slope_20",
                "cryo_panel_power_deficit_w", "cryo_panel_power_deficit_rolling_sum_20",
            ]
            if c in all_feat_cols
        ],
    },
    "training": {
        key: {
            "epochs_run": int(len(results[key]["history"])),
            "best_val_score": safe(results[key]["best_val_score"]),
            "best_val_auroc": safe(results[key]["history"]["val_auroc"].max()) if "val_auroc" in results[key]["history"] else None,
            "best_val_f1": safe(results[key]["history"]["val_f1"].max()) if "val_f1" in results[key]["history"] else None,
            "best_val_ttf_bin_macro_f1": safe(results[key]["history"]["val_ttf_bin_macro_f1"].max()) if "val_ttf_bin_macro_f1" in results[key]["history"] else None,
            "history": [{k2: safe(v2) for k2, v2 in row.items()} for row in results[key]["history"].to_dict(orient="records")],
        }
        for key in _model_keys
    },
    "test_metrics": {
        row["model_key"]: safe_record_dict(row)
        for row in comparison_df[comparison_df["split"].eq("test")].to_dict(orient="records")
    },
    "val_metrics": {
        row["model_key"]: safe_record_dict(row)
        for row in comparison_df[comparison_df["split"].eq("val")].to_dict(orient="records")
    },
    "multi_seed": {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "aggregated": {
            k: {m: {stat: safe(v) for stat, v in vals.items()} for m, vals in multi_seed_summary[k].items()}
            for k in _model_keys
        },
        "per_seed": {
            k: [{"seed": r["seed"],
                 "test_auroc": safe(r["test_metrics"].get("auroc")),
                 "test_f1": safe(r["test_metrics"].get("f1")),
                 "test_flat_cause_macro_f1_near": safe(r["test_metrics"].get("flat_cause_macro_f1_near")),
                 "test_hier_cause_macro_f1_near": safe(r["test_metrics"].get("hier_cause_macro_f1_near")),
                 "test_ttf_bin_macro_f1": safe(r["test_metrics"].get("ttf_bin_macro_f1")),
                 "test_horizon_macro_f1": safe(r["test_metrics"].get("horizon_macro_f1")),
                 "test_shared_ttf_mae_ticks": safe(r["test_metrics"].get("shared_ttf_mae_ticks")),
                 "test_shared_ttf_macro_run_mae_ticks": safe(r["test_metrics"].get("shared_ttf_macro_run_mae_ticks")),
                 "test_shared_ttf_mae_ticks_near": safe(r["test_metrics"].get("shared_ttf_mae_ticks_near")),
                 "test_shared_ttf_macro_run_mae_ticks_near": safe(r["test_metrics"].get("shared_ttf_macro_run_mae_ticks_near")),
                 "val_auroc": safe(r["val_metrics"].get("auroc")),
                 "epochs_run": int(len(r["history"]))}
                for r in all_seed_results[k]]
            for k in _model_keys
        },
    },
    "explainer": {
        key: {
            "original_prob": safe(explanations.get(key, {}).get("original_prob", np.nan)),
            "top_edges": ranking_outputs.get((key,"edges"), pd.DataFrame()).head(5).to_dict(orient="records"),
            "top_features": ranking_outputs.get((key,"features"), pd.DataFrame()).head(5).to_dict(orient="records"),
            "top_context": ranking_outputs.get((key,"context"), pd.DataFrame()).head(5).to_dict(orient="records"),
            "context_reliance": safe(context_reliance(explanations[key])) if key in explanations else None,
        }
        for key in xai_model_keys
    },
    "faithfulness": faith_df.head(500).to_dict(orient="records"),
    "xai_files": {
        "faithfulness": "faithfulness_checks.csv",
        "stability": "xai_stability_summary.csv",
        "expected_paths": "xai_expected_path_hit_rates.csv",
        "temporal_segments": "xai_temporal_segment_scores.csv",
        "context_ablation": "xai_model_b_context_ablation.csv",
        "baseline_comparison": "xai_baseline_comparison.csv",
        "png_dir": "xai_png/",
    },
}

summary_path = OUTPUT_ROOT / "run_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
log(f"run_summary.json written to {summary_path}")

# =============================================================================
# Console log
# =============================================================================
console_log_path = OUTPUT_ROOT / "train_v11_console_log.txt"
console_log_path.write_text("\n".join(_CONSOLE_LOG_BUFFER), encoding="utf-8")
log(f"train_v11_console_log.txt written to {console_log_path}")

# =============================================================================
# Experiment log  (only written when run_tag is non-empty)
# =============================================================================
if CFG.get("run_tag", ""):
    log_entry = {
        "run_tag":   CFG["run_tag"],
        "timestamp": summary["timestamp"],
        "cfg_snapshot": {k: CFG[k] for k in [
            "max_epochs","patience","learning_rate","weight_decay",
            "cause_loss_weight","family_loss_weight","ttf_bin_loss_weight","ttf_loss_weight",
            "global_ttf_loss_weight","near_ttf_loss_weight",
            "horizon_loss_weight","monotonic_horizon_loss_weight",
            "dropout","context_dropout","blocks","layers","residual_channels","skip_channels",
            "window_len","window_stride","near_failure_stride","cause_near_window_ticks",
            "ttf_near_window_ticks","ttf_refine_window_ticks","horizon_ticks","explain_epochs","xai_samples_per_cause",
        ]},
        "dataset": {
            "n_configs": summary["dataset"]["n_configs"],
            "total_runs": summary["dataset"]["total_runs"],
            "failure_rate": summary["dataset"]["failure_rate"],
        },
        "metrics": {
            key: {
                "test_auroc": summary["test_metrics"].get(key, {}).get("auroc"),
                "test_auprc": summary["test_metrics"].get(key, {}).get("auprc"),
                "test_f1": summary["test_metrics"].get(key, {}).get("f1"),
                "test_flat_cause_macro_f1_near": summary["test_metrics"].get(key, {}).get("flat_cause_macro_f1_near"),
                "test_hier_cause_macro_f1_near": summary["test_metrics"].get(key, {}).get("hier_cause_macro_f1_near"),
                "test_ttf_bin_macro_f1": summary["test_metrics"].get(key, {}).get("ttf_bin_macro_f1"),
                "test_horizon_macro_f1": summary["test_metrics"].get(key, {}).get("horizon_macro_f1"),
                "test_shared_ttf_mae_ticks": summary["test_metrics"].get(key, {}).get("shared_ttf_mae_ticks"),
                "test_shared_ttf_macro_run_mae_ticks": summary["test_metrics"].get(key, {}).get("shared_ttf_macro_run_mae_ticks"),
                "test_shared_ttf_mae_ticks_near": summary["test_metrics"].get(key, {}).get("shared_ttf_mae_ticks_near"),
                "test_shared_ttf_macro_run_mae_ticks_near": summary["test_metrics"].get(key, {}).get("shared_ttf_macro_run_mae_ticks_near"),
                "epochs_run": summary["training"].get(key, {}).get("epochs_run"),
            }
            for key in _model_keys
        },
        "multi_seed_auroc": {
            k: {"mean": safe(multi_seed_summary[k].get("auroc", {}).get("mean")),
                "std":  safe(multi_seed_summary[k].get("auroc", {}).get("std"))}
            for k in _model_keys
        },
    }
    drive_log = DRIVE_OUTPUT / "experiment_log.json"
    local_log = OUTPUT_ROOT / "experiment_log.json"
    existing = []
    for candidate in [drive_log, local_log]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    existing = json.load(f)
                break
            except Exception:
                pass
    existing.append(log_entry)
    log_json = json.dumps(existing, indent=2, default=str)
    local_log.write_text(log_json)
    log(f"experiment_log.json written locally ({len(existing)} entries) -> {local_log}")
    try:
        drive_log.write_text(log_json)
        log(f"experiment_log.json also written to Drive -> {drive_log}")
    except Exception as e:
        log(f"WARNING: could not write directly to Drive ({e}); local copy will sync below.")

# =============================================================================
# Sync to Drive
# =============================================================================
def sync_to_drive(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        d   = dst / rel
        if p.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)

log("syncing outputs to Drive ...")
sync_to_drive(OUTPUT_ROOT, DRIVE_OUTPUT)
log(f"done. outputs at {DRIVE_OUTPUT}")
log("v11 complete.")
