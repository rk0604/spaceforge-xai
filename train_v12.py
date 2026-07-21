"""
SpaceForge RT-GWN v12 -- Relational Telescope Graph WaveNet with
competing-risk hazards.

================================================================================
WHAT v12 IS
================================================================================
v12 is a structural redesign of the v11 Graph WaveNet health-management model.
It keeps the v11 data funnel (feature_engineering_v2 parquet chunks + funnel
labels on Google Drive, 13 physical nodes, config-level splits, leakage rules)
but replaces the model skeleton along three axes:

1. TYPED RELATIONS. The single adjacency of v8-v11 is split into three
   relation-specific supports that mirror the C++ simulator call graph:
     - power   : electrical flow (PowerBus drawPower / addPower)
     - heat    : thermal flow   (applyHeat / addHeatLoad / ambient coupling)
     - state   : control and state reads (setters, derating, biasing)
   Graph WaveNet's diffusion convolution runs over all three (forward and
   backward transition matrices per relation) plus one small self-adaptive
   adjacency PER RELATION, so learned shortcuts are attributable to a
   physical relation type.

2. TELESCOPE TEMPORAL ENCODING. Two parallel streams share the graph:
     - fast stream: last 64 ticks at stride 1 (same receptive field as v11)
     - slow stream: last 576 ticks (about 6 orbits) downsampled x8 to 72
       steps, so run-scale dynamics (SourceInventory depletion, BatteryThermal
       drift, CryoPanel regen cycles, eclipse periodicity) are visible natively
       instead of only through hand-made slope features.
   A learned gate fuses the two streams; an XAI "scale mask" on the two stream
   contributions yields an acute-vs-chronic score per explanation.

3. COMPETING-RISK HAZARD HEAD. The v10/v11 pile of heads (binary, cause,
   family, subtype, exact TTF, global/near TTF, cause-specific TTF, TTF bins,
   horizons + monotonic penalty) collapses into ONE discrete-time
   competing-risks head: for each future time bin (the v11 TTF_BIN_EDGES) the
   model predicts a softmax over {survive-bin, cause_1 .. cause_C}. Everything
   v11 reports falls out of this single calibrated object:
     - survival curve S(k)            -> expected TTF (within 240 ticks)
     - horizon probabilities 1 - S(h) -> monotone BY CONSTRUCTION
     - per-bin event mass             -> TTF-bin prediction (12-way incl far)
     - cause posterior                -> cause diagnosis
     - terminal risk 1 - S(K)         -> binary early-warning score
   Alive runs and far-from-failure windows are handled as right-censored
   observations in the likelihood; no has_cause / has_exact_ttf masking
   gymnastics are needed. A small auxiliary exact-TTF regressor (distance
   weighted, all pre-failure windows) covers the beyond-240-tick range.

ARCHITECTURE (per model A/B, per seed)

    x_fast (B,N,F,64)  --per-node encoders-->  (B,d,N,64)  --GatedTCN+RelGCN x6--+
                                                                                  +--> gate fusion --> heads
    x_slow (B,N,F,72)  --same encoders------>  (B,d,N,72)  --GatedTCN+RelGCN x6--+
                        (slow = 576 ticks, stride 8, edge-padded on the left)

    heads: hazard logits (B, K bins, 1+C causes)   [competing risks]
           aux exact TTF (B,)                      [softplus, ticks/240]

  Per-node encoders consume each node's TRUE feature count (no padding enters
  the network), so GNNExplainer feature masks live in each node's raw feature
  space and never spend mass on pad dimensions.

================================================================================
I/O CONTRACT (mirrors train_v11.py)
================================================================================
Inputs (Google Drive, same layout as v11 / feature_engineering_v2 outputs):
  CFG["drive_root"]/all_runs_features_model_ready_chunks/part_*.parquet
  CFG["drive_root"]/streaming_funnel_outputs/run_funnel_df.csv   (fallbacks below)

Outputs (local work dir, synced to CFG["drive_output_root"] at the end):
  train_v12_console_log.txt
  run_summary.json
  model_comparison.csv
  training_history_{model}_{seed}.csv
  per_class_cause_metrics.csv
  per_horizon_metrics.csv
  ttf_bin_metrics.csv
  ttf_regression_metrics.csv
  hazard_calibration.csv
  survival_curves_sample.csv
  adaptive_adjacency_{relation}_{model}_{seed}.csv
  xai_relation_edge_masks.csv
  xai_feature_masks.csv
  xai_acute_chronic.csv
  xai_faithfulness_v12.csv
  model_{model}_s{seed}.pt

Run on Colab:
    !python train_v12.py
Optional CFG override (used by the runner notebook):
    export SF_V12_CFG_JSON=/path/to/overrides.json
"""

# =============================================================================
# CFG -- all tunable knobs live here
# =============================================================================
CFG = {
    # data (same Drive layout as v11; point drive_root at the new 100-config
    # export produced by feature_engineering_v2)
    "drive_mount":       "/content/gdrive",
    "drive_root":        "/content/gdrive/MyDrive/SpaceForgeData/spaceforge-cleaned2/sf-cleaned-2",
    "drive_output_root": "/content/gdrive/MyDrive/SpaceForgeData/spaceforge-cleaned2/rtgwn_xai_outputs_v12",
    "local_work_root":   "/content/spaceforge_rtgwn_work_v12",

    # windowing (fast stream identical to v11)
    "window_len":        64,
    "window_stride":     5,
    "near_failure_stride": 1,
    "near_failure_stride_window_ticks": 120.0,
    "ttf_norm_ticks":    240.0,
    "max_windows_per_split": None,   # e.g. 2000 for a smoke test

    # telescope slow stream
    "slow_lookback_ticks": 576,      # about 6 orbits at ~93 ticks/orbit
    "slow_stride":         8,        # 576 / 8 = 72 steps

    # model
    "hidden_channels":    40,        # d: shared per-node embedding channels
    "skip_channels":      80,
    "end_channels":       128,
    "kernel_size":        2,
    "fast_layers":        6,         # receptive field 64 on the fast stream
    "slow_layers":        6,         # receptive field 64 steps = 512 ticks
    "dropout":            0.18,
    "graph_order":        2,
    "adaptive_adj":       True,
    "node_embedding_dim": 10,        # per relation, so 3 x (2 x N x 10)
    "context_dropout":    0.40,

    # training
    "seed":               42,
    "batch_size":         128,
    "max_epochs":         40,
    "patience":           10,
    "learning_rate":      1e-3,
    "weight_decay":       1e-4,
    "hazard_loss_weight": 1.00,
    "aux_ttf_loss_weight":0.50,
    "aux_ttf_tau_ticks":  60.0,      # distance weight exp(-ttf/tau)
    "cause_hazard_weight_cap": 3.0,  # cap on rare-cause upweighting in the NLL

    # composite validation score (v11-compatible weighting)
    "score_binary_auroc_weight": 1.00,
    "score_cause_f1_weight":     0.50,
    "score_horizon_f1_weight":   0.50,
    "score_ttf_bin_f1_weight":   0.50,
    "score_ttf_overall_mae_weight": 0.0020,
    "score_ttf_near_mae_weight":    0.0040,

    # evaluation
    "near_ttf_eval_ticks": 60.0,
    "cause_eval_window_ticks": 120.0,
    "horizon_threshold":   0.50,
    "calibration_bins":    10,

    # explainer
    "run_xai":             True,
    "explain_epochs":      150,
    "explain_lr":          0.05,
    "edge_size_penalty":   0.005,
    "edge_entropy_penalty":0.10,
    "feature_size_penalty":0.010,
    "feature_entropy_penalty": 0.10,
    "temporal_size_penalty":   0.005,
    "temporal_entropy_penalty":0.05,
    "scale_size_penalty":      0.010,
    "xai_samples_per_cause":   8,
    "xai_top_k":               10,

    # ablation / multi-seed
    "seeds":         [42, 123, 456],
    "model_keys":    ["A", "B"],     # A = subsystem only, B = + safe context

    # experiment log
    "run_tag": "R12-rtgwn-relational-telescope-hazard",
}

import os as _os
import json as _json
if _os.environ.get("SF_V12_CFG_JSON"):
    try:
        with open(_os.environ["SF_V12_CFG_JSON"], "r", encoding="utf-8") as _f:
            _overrides = _json.load(_f)
        CFG.update(_overrides)
        print(f"[cfg] applied {len(_overrides)} overrides from SF_V12_CFG_JSON")
    except Exception as _exc:
        print(f"[cfg] WARNING: failed to load SF_V12_CFG_JSON: {_exc}")

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
import re as _re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
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
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =============================================================================
# Logging helpers
# =============================================================================
_CONSOLE_LOG_LINES: List[str] = []

def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _CONSOLE_LOG_LINES.append(line)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(CFG["seed"])
log(f"device: {DEVICE}")

# =============================================================================
# Paths and staging (identical contract to v11)
# =============================================================================
try:
    if Path(CFG["drive_mount"]).exists():
        # Already mounted (normally by the parent notebook). Never call
        # drive.mount() again here: when this script runs as a subprocess it
        # has no interactive auth channel, so a mount attempt can hang the
        # whole training run with no output.
        log(f"Drive already mounted at {CFG['drive_mount']}")
    else:
        from google.colab import drive as _gdrive
        _gdrive.mount(CFG["drive_mount"])
except Exception as exc:
    log(f"Drive mount skipped: {exc}")

CLEAN_ROOT = Path(CFG["drive_root"])
if not CLEAN_ROOT.exists() and str(CLEAN_ROOT).startswith("/content/gdrive"):
    _alt = Path(str(CLEAN_ROOT).replace("/content/gdrive", "/content/drive", 1))
    if _alt.exists():
        log(f"drive_root fallback: using {_alt}")
        CLEAN_ROOT = _alt

DRIVE_OUTPUT = Path(CFG["drive_output_root"])
if not DRIVE_OUTPUT.parent.exists() and str(DRIVE_OUTPUT).startswith("/content/gdrive"):
    _alt = Path(str(DRIVE_OUTPUT).replace("/content/gdrive", "/content/drive", 1))
    if _alt.parent.exists():
        DRIVE_OUTPUT = _alt
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

LOCAL_WORK   = Path(CFG["local_work_root"])
LOCAL_DATA   = LOCAL_WORK / "data"
OUTPUT_ROOT  = LOCAL_WORK / "outputs"
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def first_existing(paths, label):
    for p in paths:
        if p.exists():
            log(f"{label} found: {p}")
            return p
    raise FileNotFoundError(f"No {label} path found. Checked:\n" + "\n".join(str(p) for p in paths))

def stage(src: Path) -> Path:
    dst = LOCAL_DATA / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        log(f"staging {src.name} to local disk ...")
        shutil.copy2(src, dst)
    else:
        log(f"using cached local {src.name}")
    return dst

def stage_chunks(src_dir: Path) -> Path:
    """Merge the chunked parquet directory into one local file with a unified
    schema (per-chunk downcast means the same column can be int64 in one chunk
    and float64 in another)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks = sorted(src_dir.glob("part_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No part_*.parquet files in {src_dir}")
    dst = LOCAL_DATA / "all_runs_features_model_ready.parquet"
    if dst.exists():
        log(f"using cached merged parquet ({len(chunks)} chunks -> {dst})")
        return dst
    log(f"merging {len(chunks)} chunks from {src_dir} ...")

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
                if pa.types.is_floating(existing) or pa.types.is_floating(t):
                    col_types[field.name] = pa.float64()
                elif pa.types.is_integer(existing) and pa.types.is_integer(t):
                    col_types[field.name] = pa.int64()
    unified = pa.schema([pa.field(n, col_types[n]) for n in col_order])
    log(f"  unified schema: {len(unified)} columns")

    writer = None
    total_rows = 0
    try:
        for i, chunk in enumerate(chunks):
            dfc = pd.read_parquet(chunk)
            table = pa.Table.from_pandas(dfc, preserve_index=False)
            del dfc
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
                log(f"  merged {i+1}/{len(chunks)} chunks ({total_rows:,} rows) ...")
    finally:
        if writer:
            writer.close()
    log(f"merged {total_rows:,} rows -> {dst}")
    return dst

CHUNKS_DIR = CLEAN_ROOT / "all_runs_features_model_ready_chunks"
if CHUNKS_DIR.exists() and any(CHUNKS_DIR.glob("part_*.parquet")):
    FEATURE_PATH = stage_chunks(CHUNKS_DIR)
else:
    FEATURE_PATH = stage(first_existing([
        CLEAN_ROOT / "all_runs_features_after_stall_cut.parquet",
        CLEAN_ROOT / "all_runs_features_after_stall_cut_df.parquet",
        CLEAN_ROOT / "all_runs_features_df.parquet",
    ], "features"))

LABEL_PATH = stage(first_existing([
    CLEAN_ROOT / "streaming_funnel_outputs" / "run_funnel_df.csv",
    CLEAN_ROOT / "run_funnel_df.csv",
    CLEAN_ROOT / "summaries" / "run_funnel_df.csv",
], "labels"))

log(f"features: {FEATURE_PATH}")
log(f"labels:   {LABEL_PATH}")

# =============================================================================
# Column-filtered feature load (13-node allowlist, v11-compatible)
# =============================================================================
_NODE_COL_PREFIXES = (
    "orbit__", "orbit_", "t_orbit", "theta_", "altitude", "sunlight", "eclipse",
    "solar_array__", "solararray__", "solar_",
    "battery__", "battery_",
    "power_bus__", "powerbus__", "power_bus_", "total_requested_power", "total_delivered_power",
    "heater_bank__", "heaterbank__", "heater_bank_",
    "effusion_cell__", "effusioncell__", "effusion_", "flux_ratio",
    "substrate__", "substrate_",
    "simulation_engine__", "simulationengine__", "simulation_",
    "array_gimbal__", "arraygimbal__", "array_gimbal_", "arraygimbal_",
    "battery_thermal__", "batterythermal__", "battery_thermal_", "batterythermal_",
    "cryo_panel__", "cryopanel__", "cryo_panel_", "cryopanel_",
    "radiator__", "radiator_",
    "source_inventory__", "sourceinventory__", "source_inventory_", "sourceinventory_",
)
_CONTEXT_PREFIXES = ("process_state__", "processstate__", "schedule_state__", "schedulestate__")
_META_COLS = {
    "run_id", "run_id_label", "config_key", "config_label", "config_id", "config_name",
    "config_label_from_funnel", "canonical_tick", "tick", "simulation_tick",
}

import pyarrow as _pa
import pyarrow.parquet as _pq

_pq_schema = _pq.read_schema(FEATURE_PATH)
def _want(c):
    cl = c.lower()
    return (c in _META_COLS
            or any(cl.startswith(p) for p in _NODE_COL_PREFIXES)
            or any(cl.startswith(p) for p in _CONTEXT_PREFIXES))
_keep_cols = [c for c in _pq_schema.names if _want(c)]
log(f"selecting {len(_keep_cols)}/{len(_pq_schema.names)} columns ...")

_pf = _pq.ParquetFile(FEATURE_PATH)
_nrows = _pf.metadata.num_rows
_str_cols = [
    c for c in _keep_cols
    if _pa.types.is_string(_pq_schema.field(c).type)
    or _pa.types.is_large_string(_pq_schema.field(c).type)
]
_num_cols = [c for c in _keep_cols if c not in _str_cols]

_num_arr2d = np.empty((_nrows, len(_num_cols)), dtype=np.float32)
_str_arrs  = {c: np.empty(_nrows, dtype=object) for c in _str_cols}
_num_idx   = {c: j for j, c in enumerate(_num_cols)}

_row = 0
for _batch in _pq.ParquetFile(FEATURE_PATH).iter_batches(batch_size=250_000, columns=_keep_cols):
    _chunk = _batch.to_pandas()
    _n = len(_chunk)
    for c in _num_cols:
        _num_arr2d[_row:_row + _n, _num_idx[c]] = _chunk[c].values
    for c in _str_cols:
        _str_arrs[c][_row:_row + _n] = _chunk[c].values
    del _chunk
    _row += _n
    log(f"  loaded {_row:,} rows ...")
gc.collect()

features_df = pd.DataFrame(_num_arr2d, columns=_num_cols)
del _num_arr2d
for _sc in _str_cols:
    features_df[_sc] = _str_arrs[_sc]
del _str_arrs
gc.collect()

run_funnel_df = pd.read_csv(LABEL_PATH)
log(f"features shape: {features_df.shape}")
log(f"labels shape:   {run_funnel_df.shape}")

# =============================================================================
# Label merge (v11-compatible)
# =============================================================================
NO_FAILURE_REASONS = {
    "", "nan", "none", "null", "no_failure", "no failure",
    "nofailure", "success", "finished", "complete", "completed",
    "ok", "passed", "pass",
}

def pick_col(dfx, candidates, required=True):
    lower = {str(c).lower(): c for c in dfx.columns}
    for c in candidates:
        if c in dfx.columns:
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
CONFIG_COL = pick_col(features_df, ["config_label","config_id","config","config_name"])
TICK_COL   = pick_col(features_df, ["canonical_tick","tick","simulation_tick"])

_keep_str = {RUN_ID_COL, CONFIG_COL, "run_id", "config_id", "config_label", "config_name"}
_drop_str = [c for c in features_df.select_dtypes("object").columns if c not in _keep_str]
if _drop_str:
    features_df.drop(columns=_drop_str, inplace=True)
    gc.collect()

features_df["canonical_tick"] = pd.to_numeric(features_df[TICK_COL], errors="coerce")
features_df[RUN_ID_COL] = features_df[RUN_ID_COL].astype(str)
features_df[CONFIG_COL] = features_df[CONFIG_COL].astype(str)
features_df["_key"]     = features_df[RUN_ID_COL].map(clean_key)

label_like_cols = [
    "funnel_failed","funnel_failure_reason","funnel_failure_tick",
    "funnel_failure_step","funnel_failure_group","funnel_failure_label",
    "coarse_cause","y_fail_run","cause_idx","run_id_label",
]
features_df.drop(columns=[c for c in label_like_cols if c in features_df.columns], inplace=True)

label_run_id_col = pick_col(run_funnel_df, ["run_id"])
label_config_col = pick_col(run_funnel_df, ["config_label","config_id","config","config_name"], required=False)
label_failed_col = pick_col(run_funnel_df, ["funnel_failed","run_failed","failed","has_failure","is_failure","y_fail_run","y_fail"], required=False)
label_reason_col = pick_col(run_funnel_df, ["funnel_failure_reason","failure_reason","final_failure_reason","reason","failure_type","coarse_cause"], required=False)
label_tick_col   = pick_col(run_funnel_df, ["funnel_failure_tick","failure_tick","first_failure_tick","cutoff_tick","fail_tick"], required=False)

rl = run_funnel_df.copy()
rl[label_run_id_col] = rl[label_run_id_col].astype(str)
rl["_key"] = rl[label_run_id_col].map(clean_key)
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
rl = rl[["_key","lbl_failed","lbl_reason","lbl_tick"]].drop_duplicates("_key")

_rl_idx = rl.set_index("_key")
for _lbl_col in ["lbl_failed","lbl_reason","lbl_tick"]:
    features_df[_lbl_col] = features_df["_key"].map(_rl_idx[_lbl_col])
del _rl_idx, rl
gc.collect()

df = features_df
df["run_id"]     = df[RUN_ID_COL].astype(str)
df["config_key"] = df[CONFIG_COL].astype(str)
df["funnel_failed"]         = pd.to_numeric(df["lbl_failed"], errors="coerce").fillna(0).astype(int)
df["funnel_failure_tick"]   = pd.to_numeric(df["lbl_tick"],   errors="coerce")
df["funnel_failure_reason"] = df["lbl_reason"].fillna("no_failure").astype(str)
_reason_norm = df["funnel_failure_reason"].astype(str).str.strip().str.lower()
df.loc[_reason_norm.isin(NO_FAILURE_REASONS) & df["funnel_failure_tick"].isna(), "funnel_failed"] = 0
df.loc[df["funnel_failed"].eq(0), "funnel_failure_reason"] = "no_failure"
log(f"merged df: {df.shape}  runs={df['run_id'].nunique()}  configs={df['config_key'].nunique()}")

# =============================================================================
# Coarse causes and run meta (v11-compatible)
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
      .agg(config_key=("config_key","first"),
           funnel_failed=("funnel_failed","max"),
           funnel_failure_reason=("funnel_failure_reason","first"),
           funnel_failure_tick=("funnel_failure_tick","min"),
           n_ticks=("canonical_tick","count"))
)
run_meta["coarse_cause"] = [map_coarse_cause(r, f) for r, f in
                            zip(run_meta["funnel_failure_reason"], run_meta["funnel_failed"])]
run_meta["y_fail_run"] = (run_meta["coarse_cause"] != "no_failure").astype(int)

bad_tick = run_meta["y_fail_run"].eq(1) & run_meta["funnel_failure_tick"].isna()
if bad_tick.any():
    log(f"dropping {int(bad_tick.sum())} positive runs with missing failure tick")
    valid = set(run_meta.loc[~bad_tick, "run_id"])
    df = df[df["run_id"].isin(valid)].copy()
    run_meta = run_meta[~bad_tick].reset_index(drop=True)

cause_names = sorted(run_meta["coarse_cause"].unique().tolist())
cause_names = ["no_failure"] + [c for c in cause_names if c != "no_failure"]
cause_to_idx = {c: i for i, c in enumerate(cause_names)}
run_meta["cause_idx"] = run_meta["coarse_cause"].map(cause_to_idx).astype(int)

# Hazard causes = failure causes only (no_failure is the censored branch).
HAZARD_CAUSES = [c for c in cause_names if c != "no_failure"]
hazard_cause_to_idx = {c: i for i, c in enumerate(HAZARD_CAUSES)}
run_meta["hazard_cause_idx"] = run_meta["coarse_cause"].map(
    lambda c: hazard_cause_to_idx.get(c, -1)).astype(int)

log(f"runs total={len(run_meta)} failed={int(run_meta['y_fail_run'].sum())}")
log(f"cause distribution:\n{run_meta['coarse_cause'].value_counts().to_string()}")
log(f"hazard causes: {HAZARD_CAUSES}")

# =============================================================================
# 13-node feature selection (v11-compatible)
# =============================================================================
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
    "lbl_failed","lbl_reason","lbl_tick","_key",
}

def norm_col(c): return _re.sub(r"[^a-zA-Z0-9]+","_",str(c)).strip("_").lower()
def is_numeric(dfx, col):
    return col not in ID_AND_LABEL_COLS and (
        pd.api.types.is_numeric_dtype(dfx[col]) or pd.to_numeric(dfx[col],errors="coerce").notna().any())
def is_leakage(col): return any(_re.search(p, norm_col(col)) for p in LEAKAGE_PATTERNS)

def find_safe_context_columns(all_cols):
    process_bases  = {norm_col(x) for x in PROCESS_SAFE_BASE}
    schedule_bases = {norm_col(x) for x in SCHEDULE_SAFE_BASE}
    proc, sched = [], []
    for col in all_cols:
        nc = norm_col(col)
        tails = {nc, "_".join(nc.split("_")[-1:]), "_".join(nc.split("_")[-2:]),
                 "_".join(nc.split("_")[-3:]), "_".join(nc.split("_")[-4:])}
        if (nc.startswith("process_state_") or nc.startswith("processstate_")) and tails & process_bases:
            proc.append(col)
        if (nc.startswith("schedule_state_") or nc.startswith("schedulestate_")) and tails & schedule_bases:
            sched.append(col)
    return sorted(proc), sorted(sched)

def select_node_columns(dfx):
    node_cols = {n: [] for n in PHYSICAL_NODES}
    for col in dfx.columns:
        if col in ID_AND_LABEL_COLS or is_leakage(col) or not is_numeric(dfx, col):
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
context_cols_b = sorted({c for c in proc_cols + sched_cols
                         if c not in ID_AND_LABEL_COLS and not is_leakage(c) and is_numeric(df, c)})

log("node feature counts:")
for n_ in PHYSICAL_NODES:
    log(f"  {n_:20s} {len(node_cols_by_node[n_])}")
log(f"safe context cols (Model B): {len(context_cols_b)}")

_empty_nodes = [n_ for n_ in PHYSICAL_NODES if len(node_cols_by_node[n_]) == 0]
if _empty_nodes:
    log(f"WARNING: nodes with ZERO matched columns: {_empty_nodes}")
    log("  -> check that feature_engineering_v2 ran on the new-node CSVs and")
    log("     that the column prefixes match NODE_PATTERNS.")

# =============================================================================
# Config-level stratified split (v11-compatible)
# =============================================================================
def split_configs(run_meta_df, seed):
    cfg_cause = (
        run_meta_df[run_meta_df["coarse_cause"] != "no_failure"]
        .groupby("config_key")["coarse_cause"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"coarse_cause": "dominant_cause"})
    )
    cfg_lvl = run_meta_df.groupby("config_key", as_index=False).agg(
        any_failure=("y_fail_run", "max"), n_runs=("run_id", "nunique"))
    cfg_lvl = cfg_lvl.merge(cfg_cause, on="config_key", how="left")
    cfg_lvl["dominant_cause"] = cfg_lvl["dominant_cause"].fillna("no_failure")

    cause_counts = cfg_lvl["dominant_cause"].value_counts()
    cfg_lvl["strat_label"] = cfg_lvl["dominant_cause"].apply(
        lambda c: c if cause_counts[c] >= 6 else "other")

    configs = cfg_lvl["config_key"].tolist()
    strat   = cfg_lvl["strat_label"].tolist()
    strat   = strat if len(set(strat)) > 1 and min(pd.Series(strat).value_counts()) >= 2 else None
    train_c, temp_c = train_test_split(configs, test_size=0.30, random_state=seed, stratify=strat)

    temp_df = cfg_lvl[cfg_lvl["config_key"].isin(temp_c)]
    strat2  = temp_df["strat_label"].tolist()
    strat2  = strat2 if len(set(strat2)) > 1 and min(pd.Series(strat2).value_counts()) >= 2 else None
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=seed, stratify=strat2)

    train_set, val_set, test_set = set(train_c), set(val_c), set(test_c)
    rng = np.random.default_rng(seed)
    for cause in [c for c in cause_counts.index if c != "no_failure"]:
        cause_cfg = cfg_lvl[cfg_lvl["dominant_cause"] == cause]["config_key"].tolist()
        for target_set, other_sets in [(test_set, [train_set, val_set]),
                                       (val_set,  [train_set])]:
            if not any(c in target_set for c in cause_cfg):
                candidates = [c for c in cause_cfg if any(c in s for s in other_sets)]
                if candidates:
                    chosen = rng.choice(candidates)
                    for s in other_sets:
                        s.discard(chosen)
                    target_set.add(chosen)
                    log(f"  [split] forced {cause} config {chosen!r}")
    return train_set, val_set, test_set

TRAIN_CONFIGS, VAL_CONFIGS, TEST_CONFIGS = split_configs(run_meta, CFG["seed"])

def split_name(cfg_key):
    if cfg_key in TRAIN_CONFIGS: return "train"
    if cfg_key in VAL_CONFIGS:   return "val"
    if cfg_key in TEST_CONFIGS:  return "test"
    return "unknown"

run_meta["split"] = run_meta["config_key"].map(split_name)
for s_ in ["train","val","test"]:
    part = run_meta[run_meta["split"].eq(s_)]
    log(f"  {s_:6s} configs={part['config_key'].nunique()} runs={len(part)} failures={int(part['y_fail_run'].sum())}")

# =============================================================================
# Hazard bins and window index
# =============================================================================
WINDOW_LEN    = int(CFG["window_len"])
WINDOW_STRIDE = int(CFG["window_stride"])
NEAR_STRIDE   = int(CFG["near_failure_stride"])
NEAR_STRIDE_TICKS = float(CFG["near_failure_stride_window_ticks"])
TTF_NORM      = float(CFG["ttf_norm_ticks"])
SLOW_LOOKBACK = int(CFG["slow_lookback_ticks"])
SLOW_STRIDE   = int(CFG["slow_stride"])
SLOW_STEPS    = SLOW_LOOKBACK // SLOW_STRIDE

# Bin edges shared with v10/v11 TTF bins; horizons are exactly the upper edges,
# so horizon probabilities are the hazard CDF evaluated at the edges.
TTF_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0]
N_HAZARD_BINS = len(TTF_BIN_EDGES) - 1                      # 11 event bins
HORIZON_TICKS = TTF_BIN_EDGES[1:]                           # 11 horizons
BIN_MIDPOINTS = [(TTF_BIN_EDGES[i] + TTF_BIN_EDGES[i+1]) / 2.0 for i in range(N_HAZARD_BINS)]
TTF_BIN_NAMES = [f"ttf_{int(TTF_BIN_EDGES[i])}_{int(TTF_BIN_EDGES[i+1])}" for i in range(N_HAZARD_BINS)] + ["ttf_beyond_240"]
TTF_FAR_BIN_IDX = N_HAZARD_BINS                              # index of the censored class

def ttf_to_event_bin(ttf_ticks):
    """Return the event bin index 0..K-1 for a failure inside the horizon, or
    -1 for censored (alive, or failure beyond the last edge)."""
    if pd.isna(ttf_ticks) or float(ttf_ticks) > TTF_BIN_EDGES[-1]:
        return -1
    t = max(float(ttf_ticks), 0.0)
    for i in range(N_HAZARD_BINS):
        if TTF_BIN_EDGES[i] <= t <= TTF_BIN_EDGES[i + 1]:
            return i
    return -1

run_meta_lut = run_meta.set_index("run_id").to_dict(orient="index")

def make_window_index(dfx, lut, window_len, stride):
    """v12 window index.

    Columns consumed by the hazard likelihood:
      event_bin  : 0..K-1 when the run fails within 240 ticks of window end
                   (bin of the TTF), else -1 (right-censored at 240)
      event_cause: hazard-cause index for the event, -1 when censored
      ttf_ticks  : exact TTF (nan for alive runs)
      aux_weight : exp(-ttf / tau) distance weight for the auxiliary exact-TTF
                   regressor (0 for alive runs)
    Near-failure windows (ttf <= near_failure_stride_window_ticks) are cut at
    stride 1 for dense terminal coverage, like v11.
    """
    tau = float(CFG["aux_ttf_tau_ticks"])
    rows = []
    for run_id, g in dfx.groupby("run_id", sort=False):
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
        hz_cause  = int(meta.get("hazard_cause_idx", -1))
        coarse    = meta.get("coarse_cause", "no_failure")

        ticks = g_sorted["canonical_tick"].to_numpy(dtype=float)
        end_pos = window_len - 1
        while end_pos < n:
            end_tick = float(ticks[end_pos])
            if y_fail == 1 and (pd.isna(fail_tick) or end_tick >= float(fail_tick)):
                break
            if y_fail == 1:
                ttf_ticks = max(float(fail_tick) - end_tick, 0.0)
                event_bin = ttf_to_event_bin(ttf_ticks)
                aux_w     = math.exp(-ttf_ticks / tau)
            else:
                ttf_ticks = np.nan
                event_bin = -1
                aux_w     = 0.0
            rows.append(dict(
                run_id=run_id,
                config_key=meta.get("config_key"),
                split=split,
                start_pos=int(end_pos - window_len + 1),
                end_pos=int(end_pos),
                end_tick=end_tick,
                failure_tick=fail_tick,
                y_fail=y_fail,
                cause_idx=cause_idx,
                coarse_cause=coarse,
                event_bin=int(event_bin),
                event_cause=int(hz_cause if event_bin >= 0 else -1),
                ttf_ticks=ttf_ticks,
                aux_weight=float(aux_w),
            ))
            near = (y_fail == 1 and not pd.isna(ttf_ticks) and ttf_ticks <= NEAR_STRIDE_TICKS)
            end_pos += NEAR_STRIDE if near else stride

    out = pd.DataFrame(rows)
    max_win = CFG["max_windows_per_split"]
    if max_win and len(out) > 0:
        rng = np.random.default_rng(CFG["seed"])
        parts = []
        for s, part in out.groupby("split", sort=False):
            if len(part) > max_win:
                idx = rng.choice(part.index.to_numpy(), size=max_win, replace=False)
                parts.append(part.loc[idx])
            else:
                parts.append(part)
        out = pd.concat(parts, ignore_index=True)
    return out.reset_index(drop=True)

log("building window index ...")
window_index_df = make_window_index(df, run_meta_lut, WINDOW_LEN, WINDOW_STRIDE)
if window_index_df.empty:
    raise ValueError("No windows created -- check WINDOW_LEN vs canonical_tick range")

bad = window_index_df[
    window_index_df["y_fail"].eq(1)
    & pd.to_numeric(window_index_df["end_tick"], errors="coerce").ge(
        pd.to_numeric(window_index_df["failure_tick"], errors="coerce"))
]
assert len(bad) == 0, "leakage: windows ending at/after failure tick"

log("window counts:")
for s_ in ["train","val","test"]:
    w = window_index_df[window_index_df["split"].eq(s_)]
    log(f"  {s_:6s} windows={len(w)} positives={int(w['y_fail'].sum())} "
        f"events_within_240={int((w['event_bin'] >= 0).sum())}")

# Per-cause NLL weights: rarer hazard causes get a bounded upweight.
_train_events = window_index_df[(window_index_df["split"].eq("train")) & (window_index_df["event_bin"] >= 0)]
_ev_counts = _train_events["event_cause"].value_counts().reindex(range(len(HAZARD_CAUSES)), fill_value=1)
_cause_w = len(_train_events) / (max(len(HAZARD_CAUSES),1) * _ev_counts.values.astype(float))
_cause_w = np.clip(_cause_w, 1.0 / CFG["cause_hazard_weight_cap"], CFG["cause_hazard_weight_cap"])
HAZARD_CAUSE_WEIGHTS = torch.tensor(_cause_w, dtype=torch.float32)
log("hazard cause weights:")
for i, cname in enumerate(HAZARD_CAUSES):
    log(f"  {cname:25s} weight={float(HAZARD_CAUSE_WEIGHTS[i]):.3f}")

# =============================================================================
# Normalization + per-run frames
# =============================================================================
@dataclass
class ModelSpec:
    name: str
    node_cols_by_node: Dict[str, List[str]]
    context_cols: List[str]
    max_node_features: int
    node_feature_counts: List[int]

def build_spec(name, node_cols, ctx_cols):
    max_f  = max(max(len(c), 1) for c in node_cols.values())
    counts = [max(len(node_cols.get(n_, [])), 1) for n_ in PHYSICAL_NODES]
    return ModelSpec(name=name,
                     node_cols_by_node={k: list(v) for k, v in node_cols.items()},
                     context_cols=list(ctx_cols),
                     max_node_features=max_f,
                     node_feature_counts=counts)

MODEL_A_SPEC = build_spec("subsystem_only",              node_cols_by_node, [])
MODEL_B_SPEC = build_spec("subsystem_plus_safe_context", node_cols_by_node, context_cols_b)

all_feat_cols = set()
for cols in node_cols_by_node.values():
    all_feat_cols.update(cols)
all_feat_cols.update(context_cols_b)

keep_cols = ["run_id","canonical_tick"] + sorted(all_feat_cols)
keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
df_model  = df[keep_cols].copy()
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
        if not m.any():
            continue
        c = v[m]; count += c.size; total += float(c.sum()); sq += float(np.square(c).sum())
    if count == 0:
        return 0.0, 1.0
    mean = total / count
    var  = max((sq - total*total/count) / (count-1) if count > 1 else 0.0, 0.0)
    std  = math.sqrt(var)
    return float(mean), float(std) if (np.isfinite(std) and std >= 1e-8) else 1.0

log("computing normalizer ...")
normalizer = {col: compute_normalizer(col) for col in sorted(all_feat_cols)}

def normalize_array(values, col):
    mean, std = normalizer.get(col, (0.0, 1.0))
    out = (values.astype(np.float32) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# =============================================================================
# Dataset: fast window + telescope slow window from the same run cache
# =============================================================================
class SpaceForgeTelescopeDataset(Dataset):
    """Serves (x_fast, x_slow, context_fast) per window.

    x_fast: (N, F, window_len)   last 64 ticks at stride 1
    x_slow: (N, F, SLOW_STEPS)   last slow_lookback ticks at slow_stride,
                                 edge-padded on the left with the earliest
                                 available frame when the run is younger than
                                 the lookback (physically: the state before the
                                 run started is its initial state)
    """
    def __init__(self, window_index, run_frames_, spec):
        self.wi   = window_index.reset_index(drop=True).copy()
        self.spec = spec
        self.run_ids   = self.wi["run_id"].astype(str).to_numpy()
        self.starts    = self.wi["start_pos"].astype(int).to_numpy()
        self.ends      = self.wi["end_pos"].astype(int).to_numpy()
        self.end_ticks = self.wi["end_tick"].astype(float).to_numpy()
        self.y_fail    = self.wi["y_fail"].astype(np.float32).to_numpy()
        self.cause_idx = self.wi["cause_idx"].astype(int).to_numpy()
        self.event_bin = self.wi["event_bin"].astype(int).to_numpy()
        self.event_cause = self.wi["event_cause"].astype(int).to_numpy()
        self.ttf_ticks = pd.to_numeric(self.wi["ttf_ticks"], errors="coerce").to_numpy(dtype=np.float32)
        self.aux_weight = self.wi["aux_weight"].astype(np.float32).to_numpy()
        self.cache = {}
        for run_id in sorted(set(self.run_ids)):
            g     = run_frames_[run_id]
            n_n   = len(PHYSICAL_NODES)
            max_f = spec.max_node_features
            t_len = len(g)
            x = np.zeros((n_n, max_f, t_len), dtype=np.float32)
            for ni, node in enumerate(PHYSICAL_NODES):
                for fi, col in enumerate(spec.node_cols_by_node.get(node, [])[:max_f]):
                    if col in g.columns:
                        x[ni, fi, :] = normalize_array(
                            pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32), col)
            c_cols = spec.context_cols
            ctx = np.zeros((len(c_cols), t_len), dtype=np.float32) if c_cols else np.zeros((0, t_len), dtype=np.float32)
            for ci, col in enumerate(c_cols):
                if col in g.columns:
                    ctx[ci, :] = normalize_array(
                        pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32), col)
            self.cache[run_id] = {"x": torch.from_numpy(x), "context": torch.from_numpy(ctx)}
        gc.collect()

    def __len__(self):
        return len(self.wi)

    def _slow_slice(self, x_full, end_pos):
        """Left-edge-padded strided slice ending at end_pos (inclusive)."""
        t_len = x_full.shape[-1]
        idx = torch.arange(end_pos - SLOW_LOOKBACK + SLOW_STRIDE, end_pos + 1, SLOW_STRIDE)
        idx = idx.clamp(min=0, max=t_len - 1)
        return x_full[..., idx]

    def __getitem__(self, idx):
        rid   = self.run_ids[idx]
        s, e  = int(self.starts[idx]), int(self.ends[idx])
        cached = self.cache[rid]
        x_fast = cached["x"][:, :, s:e+1].contiguous()
        x_slow = self._slow_slice(cached["x"], e).contiguous()
        ctx    = cached["context"][:, s:e+1].contiguous()
        return {
            "x_fast":     x_fast,
            "x_slow":     x_slow,
            "context":    ctx,
            "y_fail":     torch.tensor(self.y_fail[idx], dtype=torch.float32),
            "cause_idx":  torch.tensor(self.cause_idx[idx], dtype=torch.long),
            "event_bin":  torch.tensor(self.event_bin[idx], dtype=torch.long),
            "event_cause":torch.tensor(self.event_cause[idx], dtype=torch.long),
            "ttf_ticks":  torch.tensor(np.nan_to_num(self.ttf_ticks[idx], nan=-1.0), dtype=torch.float32),
            "aux_weight": torch.tensor(self.aux_weight[idx], dtype=torch.float32),
            "window_id":  torch.tensor(idx, dtype=torch.long),
            "end_tick":   torch.tensor(self.end_ticks[idx], dtype=torch.float32),
        }

def make_loaders(spec, batch_size):
    ds, loaders = {}, {}
    for split in ["train", "val", "test"]:
        wi = window_index_df[window_index_df["split"].eq(split)].reset_index(drop=True)
        log(f"building {spec.name} {split} dataset: {len(wi)} windows")
        ds[split] = SpaceForgeTelescopeDataset(wi, run_frames, spec)
        if split == "train":
            # Positives are upweighted by proximity to failure; rare-cause
            # events get a bounded extra boost.
            ttf = pd.to_numeric(wi["ttf_ticks"], errors="coerce").to_numpy(dtype=float)
            w = np.ones(len(wi), dtype=np.float64)
            for i in range(len(wi)):
                if wi.loc[i, "y_fail"] == 1:
                    t = ttf[i] if np.isfinite(ttf[i]) else 240.0
                    w[i] = 1.0 + 2.0 * math.exp(-t / 120.0)
                    ec = int(wi.loc[i, "event_cause"])
                    if ec >= 0:
                        w[i] *= float(HAZARD_CAUSE_WEIGHTS[ec]) ** 0.5
            sampler = WeightedRandomSampler(torch.from_numpy(w), len(w), replacement=True)
            loaders[split] = DataLoader(ds[split], batch_size=batch_size, sampler=sampler,
                                        num_workers=0, pin_memory=torch.cuda.is_available())
        else:
            loaders[split] = DataLoader(ds[split], batch_size=batch_size, shuffle=False,
                                        num_workers=0, pin_memory=torch.cuda.is_available())
    return ds, loaders

# =============================================================================
# Typed relational graph supports
# =============================================================================
RELATIONS = ["power", "heat", "state"]

# Edge assignments mirror the C++ call graph on decoupled-sparta-copy:
#   power = PowerBus drawPower/addPower; heat = applyHeat/addHeatLoad/ambient;
#   state = setters, reads, derating, biasing, bookkeeping.
TYPED_EDGES = {
    "power": [
        ("solar_array", "power_bus"),
        ("battery", "power_bus"),
        ("power_bus", "battery"),
        ("power_bus", "heater_bank"),
        ("power_bus", "array_gimbal"),
        ("power_bus", "battery_thermal"),
        ("power_bus", "cryo_panel"),
        ("power_bus", "radiator"),
    ],
    "heat": [
        ("heater_bank", "effusion_cell"),
        ("heater_bank", "substrate"),
        ("battery_thermal", "radiator"),
        ("radiator", "battery_thermal"),
        ("cryo_panel", "radiator"),
        ("orbit", "effusion_cell"),
        ("orbit", "substrate"),
        ("orbit", "cryo_panel"),
        ("orbit", "radiator"),
    ],
    "state": [
        ("orbit", "solar_array"),
        ("orbit", "array_gimbal"),
        ("array_gimbal", "solar_array"),
        ("battery", "battery_thermal"),
        ("battery_thermal", "battery"),
        ("source_inventory", "effusion_cell"),
        ("effusion_cell", "source_inventory"),
        ("effusion_cell", "substrate"),
        ("simulation_engine", "battery"),
        ("simulation_engine", "power_bus"),
    ],
}

node_to_idx = {n_: i for i, n_ in enumerate(PHYSICAL_NODES)}
idx_to_node = {i: n_ for n_, i in node_to_idx.items()}
N_NODES = len(PHYSICAL_NODES)

def row_norm(mat):
    mat = mat.astype(np.float32)
    d = mat.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return mat / d

def build_relation_supports():
    """Two static supports per relation (forward + backward transition), in
    Graph WaveNet's diffusion-convolution style, as torch tensors."""
    supports, support_relation = [], []
    for rel in RELATIONS:
        a = np.zeros((N_NODES, N_NODES), dtype=np.float32)
        for src, dst in TYPED_EDGES[rel]:
            a[node_to_idx[src], node_to_idx[dst]] = 1.0
        supports.append(torch.from_numpy(row_norm(a)))
        support_relation.append(rel)
        supports.append(torch.from_numpy(row_norm(a.T)))
        support_relation.append(rel)
    return supports, support_relation

FIXED_SUPPORTS, SUPPORT_RELATION = build_relation_supports()
log(f"typed supports: {len(FIXED_SUPPORTS)} static (2 per relation), "
    f"edges: " + ", ".join(f"{r}={len(TYPED_EDGES[r])}" for r in RELATIONS))

# =============================================================================
# Model
# =============================================================================
class PerNodeEncoder(nn.Module):
    """One 1x1 conv per node, consuming only that node's true feature count.
    Padding never enters the network, so explainer feature masks live in each
    node's raw feature space."""
    def __init__(self, node_feature_counts, out_channels):
        super().__init__()
        self.counts = list(node_feature_counts)
        self.encoders = nn.ModuleList([
            nn.Conv2d(cnt, out_channels, kernel_size=(1, 1)) for cnt in self.counts
        ])

    def forward(self, x):
        # x: (B, N, F_max, T) -> (B, d, N, T)
        outs = []
        for i, enc in enumerate(self.encoders):
            xi = x[:, i, : self.counts[i], :].unsqueeze(2)   # (B, F_i, 1, T)
            outs.append(enc(xi))                              # (B, d, 1, T)
        return torch.cat(outs, dim=2)

class RelationalDiffusionConv(nn.Module):
    """Graph WaveNet diffusion convolution over typed supports.

    y = sum_over_supports sum_{k=1..order} W_{s,k} (A_s^k x)
    Static supports may be elementwise-masked by the explainer (edge_masks is
    a dict relation -> (N,N) mask in [0,1]). Adaptive supports (one per
    relation) are built from small node embeddings, softmax(relu(E1 E2^T)).
    """
    def __init__(self, c_in, c_out, n_supports, order, dropout):
        super().__init__()
        self.order = order
        total = c_in * (1 + n_supports * order)
        self.mlp = nn.Conv2d(total, c_out, kernel_size=(1, 1))
        self.dropout = dropout

    @staticmethod
    def _aconv(x, a):
        # x: (B, C, N, T), a: (N, N) -> propagate along nodes
        return torch.einsum("bcnt,nm->bcmt", x, a)

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            xk = x
            for _ in range(self.order):
                xk = self._aconv(xk, a)
                out.append(xk)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)

class StreamEncoder(nn.Module):
    """One Graph WaveNet stack (gated dilated causal TCN + relational
    diffusion conv per layer, residual + skip) producing a pooled skip
    embedding for the stream."""
    def __init__(self, in_channels, layers, kernel_size, n_supports, order,
                 residual_channels, dilation_channels, skip_channels, dropout):
        super().__init__()
        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.skip_convs   = nn.ModuleList()
        self.gconvs       = nn.ModuleList()
        self.bns          = nn.ModuleList()
        self.kernel_size  = kernel_size
        d = 1
        for _ in range(layers):
            self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                               kernel_size=(1, kernel_size), dilation=(1, d)))
            self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=(1, d)))
            self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
            self.gconvs.append(RelationalDiffusionConv(dilation_channels, residual_channels,
                                                       n_supports, order, dropout))
            self.bns.append(nn.BatchNorm2d(residual_channels))
            d *= 2

    def forward(self, x, supports):
        # x: (B, residual_channels, N, T)
        skip_total = None
        for i in range(len(self.filter_convs)):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate                                   # (B, C, N, T')
            s = self.skip_convs[i](x)
            s = s[..., -1:]                                   # last causal step
            skip_total = s if skip_total is None else skip_total + s
            x = self.gconvs[i](x, supports)
            x = x + residual[..., -x.shape[-1]:]
            x = self.bns[i](x)
        return skip_total                                     # (B, skip, N, 1)

class RTGWN(nn.Module):
    """Relational Telescope Graph WaveNet with a competing-risks hazard head.

    forward(x_fast, x_slow, context, masks=None) -> dict with:
      hazard_logits : (B, K, 1 + C)   per-bin softmax over survive + causes
      aux_ttf       : (B,)            softplus, units of TTF_NORM ticks
      stream_gate   : (B, 1)          learned fast-vs-slow fusion gate
    masks (all optional, used by the explainer):
      edge_masks    : {relation: (N,N) tensor in [0,1]} multiplies both static
                      supports of that relation and its adaptive support
      feature_mask  : (N, F_max) in [0,1], multiplies raw inputs of both streams
      temporal_mask : (window_len,) in [0,1], multiplies the fast stream input
      scale_mask    : (2,) in [0,1], gates [fast, slow] stream skip embeddings
    """
    def __init__(self, spec, n_causes,
                 hidden_channels, skip_channels, end_channels,
                 kernel_size, fast_layers, slow_layers,
                 dropout, graph_order, adaptive_adj, node_embedding_dim,
                 context_dim, context_dropout):
        super().__init__()
        self.spec = spec
        self.n_causes = n_causes
        self.adaptive_adj = adaptive_adj
        self.graph_order = graph_order
        self.encoder = PerNodeEncoder(spec.node_feature_counts, hidden_channels)

        n_static = len(FIXED_SUPPORTS)
        n_supports = n_static + (len(RELATIONS) if adaptive_adj else 0)
        self.register_buffer_supports()

        if adaptive_adj:
            self.adapt_e1 = nn.ParameterList([
                nn.Parameter(torch.randn(N_NODES, node_embedding_dim) * 0.1) for _ in RELATIONS])
            self.adapt_e2 = nn.ParameterList([
                nn.Parameter(torch.randn(N_NODES, node_embedding_dim) * 0.1) for _ in RELATIONS])

        self.fast = StreamEncoder(hidden_channels, fast_layers, kernel_size, n_supports,
                                  graph_order, hidden_channels, hidden_channels,
                                  skip_channels, dropout)
        self.slow = StreamEncoder(hidden_channels, slow_layers, kernel_size, n_supports,
                                  graph_order, hidden_channels, hidden_channels,
                                  skip_channels, dropout)

        self.context_dim = context_dim
        self.context_dropout = context_dropout
        if context_dim > 0:
            self.context_proj = nn.Conv1d(context_dim, skip_channels, kernel_size=1)

        # Stream fusion gate: sigmoid scalar from both pooled embeddings.
        self.gate_fc = nn.Linear(2 * skip_channels, 1)

        head_in = 2 * skip_channels + (skip_channels if context_dim > 0 else 0)
        self.end = nn.Sequential(
            nn.ReLU(),
            nn.Linear(head_in, end_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hazard_head  = nn.Linear(end_channels, N_HAZARD_BINS * (1 + n_causes))
        self.aux_ttf_head = nn.Linear(end_channels, 1)

    def register_buffer_supports(self):
        for i, s in enumerate(FIXED_SUPPORTS):
            self.register_buffer(f"support_{i}", s.clone())

    def _supports(self, edge_masks=None):
        sup = []
        for i, rel in enumerate(SUPPORT_RELATION):
            a = getattr(self, f"support_{i}")
            if edge_masks is not None and rel in edge_masks:
                m = edge_masks[rel]
                # forward supports use m, backward supports use m.T
                a = a * (m if i % 2 == 0 else m.t())
            sup.append(a)
        if self.adaptive_adj:
            for r, rel in enumerate(RELATIONS):
                adp = F.softmax(F.relu(self.adapt_e1[r] @ self.adapt_e2[r].t()), dim=1)
                if edge_masks is not None and rel in edge_masks:
                    adp = adp * edge_masks[rel]
                sup.append(adp)
        return sup

    def adaptive_adjacency(self):
        """Learned adjacency per relation for logging / plots."""
        out = {}
        if self.adaptive_adj:
            for r, rel in enumerate(RELATIONS):
                out[rel] = F.softmax(F.relu(self.adapt_e1[r] @ self.adapt_e2[r].t()),
                                     dim=1).detach().cpu().numpy()
        return out

    def forward(self, x_fast, x_slow, context=None, masks=None):
        masks = masks or {}
        feature_mask  = masks.get("feature_mask")
        temporal_mask = masks.get("temporal_mask")
        scale_mask    = masks.get("scale_mask")
        edge_masks    = masks.get("edge_masks")

        if feature_mask is not None:
            x_fast = x_fast * feature_mask.unsqueeze(0).unsqueeze(-1)
            x_slow = x_slow * feature_mask.unsqueeze(0).unsqueeze(-1)
        if temporal_mask is not None:
            x_fast = x_fast * temporal_mask.view(1, 1, 1, -1)

        supports = self._supports(edge_masks)
        h_fast = self.encoder(x_fast)                          # (B, d, N, Tf)
        h_slow = self.encoder(x_slow)                          # (B, d, N, Ts)
        s_fast = self.fast(h_fast, supports)                   # (B, skip, N, 1)
        s_slow = self.slow(h_slow, supports)

        # Pool nodes -> stream embeddings.
        e_fast = s_fast.mean(dim=2).squeeze(-1)                # (B, skip)
        e_slow = s_slow.mean(dim=2).squeeze(-1)

        if scale_mask is not None:
            e_fast = e_fast * scale_mask[0]
            e_slow = e_slow * scale_mask[1]

        gate = torch.sigmoid(self.gate_fc(torch.cat([e_fast, e_slow], dim=1)))
        fused = torch.cat([e_fast * gate, e_slow * (1.0 - gate)], dim=1)

        if self.context_dim > 0 and context is not None and context.numel() > 0:
            c = self.context_proj(context).mean(dim=-1)        # (B, skip)
            c = F.dropout(c, self.context_dropout, training=self.training)
            fused = torch.cat([fused, c], dim=1)

        h = self.end(fused)
        hazard_logits = self.hazard_head(h).view(-1, N_HAZARD_BINS, 1 + self.n_causes)
        aux_ttf = F.softplus(self.aux_ttf_head(h)).squeeze(-1)
        return {"hazard_logits": hazard_logits, "aux_ttf": aux_ttf,
                "stream_gate": gate.squeeze(-1)}

# =============================================================================
# Hazard likelihood and derived quantities
# =============================================================================
def hazard_log_probs(hazard_logits):
    """log-softmax over {survive, cause_1..C} per bin.
    Returns (logp_survive (B,K), logp_event (B,K,C))."""
    logp = F.log_softmax(hazard_logits, dim=-1)
    return logp[..., 0], logp[..., 1:]

def hazard_nll(hazard_logits, event_bin, event_cause, cause_weights=None):
    """Discrete-time competing-risks negative log likelihood.

    event_bin >= 0: survived bins 0..k-1, then event of cause c in bin k:
        -( sum_{j<k} logp_surv_j + logp_event[k, c] )
    event_bin == -1 (censored at the 240-tick horizon):
        -( sum_{j=0..K-1} logp_surv_j )
    """
    logp_surv, logp_event = hazard_log_probs(hazard_logits)   # (B,K), (B,K,C)
    B, K = logp_surv.shape
    cum = torch.cumsum(logp_surv, dim=1)                      # (B,K)
    cum_prev = F.pad(cum, (1, 0))[:, :K]                      # sum_{j<k}

    censored = event_bin.lt(0)
    k_safe = event_bin.clamp(min=0)
    c_safe = event_cause.clamp(min=0)

    ll_event = cum_prev.gather(1, k_safe.unsqueeze(1)).squeeze(1) + \
               logp_event.gather(1, k_safe.view(-1, 1, 1).expand(-1, 1, logp_event.shape[-1])
                                 ).squeeze(1).gather(1, c_safe.unsqueeze(1)).squeeze(1)
    ll_cens = cum[:, -1]
    ll = torch.where(censored, ll_cens, ll_event)

    if cause_weights is not None:
        w = torch.ones_like(ll)
        w_event = cause_weights.to(ll.device)[c_safe]
        w = torch.where(censored, w, w_event)
        return -(ll * w).mean()
    return -ll.mean()

def hazard_derivatives(hazard_logits):
    """All reported quantities from the hazard logits.

    Returns dict of numpy arrays:
      prob_fail      (B,)    1 - S(K): P(failure within 240 ticks)
      horizon_cdf    (B,K)   P(failure <= edge_k) -- monotone by construction
      event_mass     (B,K)   P(event in bin k)
      far_mass       (B,)    S(K)
      bin_pred       (B,)    argmax over 12 classes (11 bins + far)
      cause_post     (B,C)   P(cause | failure within 240)
      expected_ttf   (B,)    sum_k mid_k P(event k) + 240 S(K), capped estimate
    """
    with torch.no_grad():
        logp_surv, logp_event = hazard_log_probs(hazard_logits)
        B, K = logp_surv.shape
        cum = torch.cumsum(logp_surv, dim=1)
        S = torch.exp(cum)                                    # (B,K) survival at edge k
        S_prev = torch.exp(F.pad(cum, (1, 0))[:, :K])         # S_{k-1}, S_{-1}=1
        p_event_bin = S_prev * (1.0 - torch.exp(logp_surv))   # (B,K) total event mass in k
        p_event_cause = S_prev.unsqueeze(-1) * torch.exp(logp_event)  # (B,K,C)
        far = S[:, -1]
        cdf = 1.0 - S
        cls_mass = torch.cat([p_event_bin, far.unsqueeze(1)], dim=1)  # (B, K+1)
        bin_pred = cls_mass.argmax(dim=1)
        cause_mass = p_event_cause.sum(dim=1)                 # (B,C)
        cause_post = cause_mass / cause_mass.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mids = torch.tensor(BIN_MIDPOINTS, dtype=torch.float32, device=S.device)
        expected_ttf = (p_event_bin * mids).sum(dim=1) + far * TTF_BIN_EDGES[-1]
        return {
            "prob_fail":    (1.0 - far).cpu().numpy(),
            "horizon_cdf":  cdf.cpu().numpy(),
            "event_mass":   p_event_bin.cpu().numpy(),
            "far_mass":     far.cpu().numpy(),
            "bin_pred":     bin_pred.cpu().numpy(),
            "cause_post":   cause_post.cpu().numpy(),
            "expected_ttf": expected_ttf.cpu().numpy(),
        }

def multitask_loss(outputs, batch):
    haz = hazard_nll(outputs["hazard_logits"], batch["event_bin"],
                     batch["event_cause"], HAZARD_CAUSE_WEIGHTS)
    aux_mask = batch["y_fail"].eq(1.0) & batch["ttf_ticks"].ge(0.0)
    if aux_mask.any():
        pred = outputs["aux_ttf"][aux_mask]
        target = batch["ttf_ticks"][aux_mask] / TTF_NORM
        w = batch["aux_weight"][aux_mask].clamp(min=0.02)
        per = F.smooth_l1_loss(pred, target, reduction="none")
        aux = (per * w).sum() / w.sum().clamp(min=1e-8)
    else:
        aux = outputs["aux_ttf"].sum() * 0.0
    total = CFG["hazard_loss_weight"] * haz + CFG["aux_ttf_loss_weight"] * aux
    return total, {"hazard_nll": float(haz.detach().cpu()),
                   "aux_ttf": float(aux.detach().cpu())}

# =============================================================================
# Train / predict / summarize
# =============================================================================
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
    return {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}

def train_one_epoch(model, loader, optimizer, scaler, label, epoch):
    model.train()
    losses, hazs, auxs = [], [], []
    for batch in loader:
        batch = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(batch["x_fast"], batch["x_slow"], batch["context"])
            loss, parts = multitask_loss(out, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        hazs.append(parts["hazard_nll"])
        auxs.append(parts["aux_ttf"])
    return {"loss": float(np.mean(losses)), "hazard_nll": float(np.mean(hazs)),
            "aux_ttf": float(np.mean(auxs))}

@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    recs = []
    for batch in loader:
        batch = to_device(batch)
        with autocast_ctx():
            out = model(batch["x_fast"], batch["x_slow"], batch["context"])
        d = hazard_derivatives(out["hazard_logits"].float())
        n = len(d["prob_fail"])
        for i in range(n):
            rec = {
                "window_id":  int(batch["window_id"][i].cpu()),
                "y_fail":     float(batch["y_fail"][i].cpu()),
                "event_bin":  int(batch["event_bin"][i].cpu()),
                "event_cause":int(batch["event_cause"][i].cpu()),
                "ttf_ticks":  float(batch["ttf_ticks"][i].cpu()),
                "prob_fail":  float(d["prob_fail"][i]),
                "bin_pred":   int(d["bin_pred"][i]),
                "far_mass":   float(d["far_mass"][i]),
                "expected_ttf": float(d["expected_ttf"][i]),
                "aux_ttf_ticks": float(out["aux_ttf"][i].float().cpu()) * TTF_NORM,
                "stream_gate": float(out["stream_gate"][i].float().cpu()),
            }
            for k in range(N_HAZARD_BINS):
                rec[f"cdf_{int(HORIZON_TICKS[k])}"] = float(d["horizon_cdf"][i, k])
            for c in range(len(HAZARD_CAUSES)):
                rec[f"cause_post_{HAZARD_CAUSES[c]}"] = float(d["cause_post"][i, c])
            recs.append(rec)
    return pd.DataFrame(recs)

def true_bin_12(row):
    return int(row["event_bin"]) if int(row["event_bin"]) >= 0 else TTF_FAR_BIN_IDX

def summarize(pred, split, label):
    out = {"split": split, "model": label, "n_windows": len(pred)}
    y = pred["y_fail"].to_numpy(dtype=int)
    p = pred["prob_fail"].to_numpy(dtype=float)
    yhat = (p >= 0.5).astype(int)
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    out["ap"]    = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")
    pr, rc, f1v, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    out["precision"], out["recall"], out["f1"] = float(pr), float(rc), float(f1v)

    # Horizon metrics from the CDF (monotone by construction).
    hf1 = []
    for h in HORIZON_TICKS:
        col = f"cdf_{int(h)}"
        true_h = ((pred["y_fail"] == 1) &
                  (pred["ttf_ticks"] >= 0) &
                  (pred["ttf_ticks"] <= h)).astype(int).to_numpy()
        pred_h = (pred[col].to_numpy(dtype=float) >= CFG["horizon_threshold"]).astype(int)
        f1h = f1_score(true_h, pred_h, zero_division=0)
        out[f"horizon_{int(h)}_f1"] = float(f1h)
        hf1.append(f1h)
    out["horizon_macro_f1"] = float(np.mean(hf1))

    # 12-way TTF-bin metrics.
    tb = pred.apply(true_bin_12, axis=1).to_numpy(dtype=int)
    pb = pred["bin_pred"].to_numpy(dtype=int)
    out["ttf_bin_acc"] = float(accuracy_score(tb, pb))
    out["ttf_bin_macro_f1"] = float(f1_score(tb, pb, average="macro", zero_division=0))

    # Cause metrics near failure (posterior argmax).
    near = pred[(pred["event_bin"] >= 0) &
                (pred["ttf_ticks"] <= CFG["cause_eval_window_ticks"])]
    if len(near) > 0 and len(HAZARD_CAUSES) > 0:
        post_cols = [f"cause_post_{c}" for c in HAZARD_CAUSES]
        cp = near[post_cols].to_numpy(dtype=float).argmax(axis=1)
        ct = near["event_cause"].to_numpy(dtype=int)
        out["cause_acc_near"] = float(accuracy_score(ct, cp))
        out["cause_macro_f1_near"] = float(f1_score(ct, cp, average="macro", zero_division=0))
    else:
        out["cause_acc_near"] = float("nan")
        out["cause_macro_f1_near"] = float("nan")

    # TTF MAE: overall from the auxiliary head (all pre-failure windows),
    # near from the hazard expectation (<= near_ttf_eval_ticks).
    pos = pred[(pred["y_fail"] == 1) & (pred["ttf_ticks"] >= 0)]
    out["ttf_mae_ticks"] = float((pos["aux_ttf_ticks"] - pos["ttf_ticks"]).abs().mean()) if len(pos) else float("nan")
    near_ttf = pos[pos["ttf_ticks"] <= CFG["near_ttf_eval_ticks"]]
    out["ttf_mae_ticks_near"] = float((near_ttf["expected_ttf"] - near_ttf["ttf_ticks"]).abs().mean()) if len(near_ttf) else float("nan")

    # Binary calibration (ECE, equal-width bins on prob_fail).
    nb = int(CFG["calibration_bins"])
    edges = np.linspace(0, 1, nb + 1)
    ece, rows = 0.0, []
    for bi in range(nb):
        m = (p >= edges[bi]) & (p < edges[bi + 1] if bi < nb - 1 else p <= edges[bi + 1])
        if m.sum() > 0:
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += (m.sum() / len(p)) * abs(conf - acc)
            rows.append({"bin_lo": float(edges[bi]), "bin_hi": float(edges[bi+1]),
                         "count": int(m.sum()), "mean_prob": conf, "frac_pos": acc})
    out["ece"] = float(ece)
    out["_calibration_rows"] = rows
    return out

def validation_score(s):
    v = (CFG["score_binary_auroc_weight"] * (s.get("auroc") or 0.0)
         + CFG["score_cause_f1_weight"]   * (0.0 if math.isnan(s.get("cause_macro_f1_near", float("nan"))) else s["cause_macro_f1_near"])
         + CFG["score_horizon_f1_weight"] * s.get("horizon_macro_f1", 0.0)
         + CFG["score_ttf_bin_f1_weight"] * s.get("ttf_bin_macro_f1", 0.0))
    if not math.isnan(s.get("ttf_mae_ticks", float("nan"))):
        v -= CFG["score_ttf_overall_mae_weight"] * s["ttf_mae_ticks"]
    if not math.isnan(s.get("ttf_mae_ticks_near", float("nan"))):
        v -= CFG["score_ttf_near_mae_weight"] * s["ttf_mae_ticks_near"]
    return float(v)

# =============================================================================
# XAI: GNNExplainer-style masks on relations, raw features, time, and scale
# =============================================================================
def explain_window(model, item, target_cause=None):
    """Optimize soft masks that preserve the model's prediction (mutual
    information objective) under size + entropy sparsity penalties.

    Masks: per-relation edge masks (N,N), per-node raw feature mask (N,F_max),
    fast temporal mask (window_len,), and a 2-vector scale mask over the
    (fast, slow) streams whose optimized values give the acute-vs-chronic
    attribution.
    """
    model.eval()
    x_fast = item["x_fast"].unsqueeze(0).to(DEVICE)
    x_slow = item["x_slow"].unsqueeze(0).to(DEVICE)
    ctx    = item["context"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        base = model(x_fast, x_slow, ctx)
        base_d = hazard_derivatives(base["hazard_logits"].float())
        base_risk = float(base_d["prob_fail"][0])

    edge_logits = {rel: torch.zeros(N_NODES, N_NODES, device=DEVICE, requires_grad=True)
                   for rel in RELATIONS}
    feat_logit  = torch.zeros(N_NODES, MODEL_SPEC_ACTIVE.max_node_features,
                              device=DEVICE, requires_grad=True)
    temp_logit  = torch.zeros(WINDOW_LEN, device=DEVICE, requires_grad=True)
    scale_logit = torch.zeros(2, device=DEVICE, requires_grad=True)

    params = list(edge_logits.values()) + [feat_logit, temp_logit, scale_logit]
    opt = torch.optim.Adam(params, lr=CFG["explain_lr"])

    def entropy(m):
        m = m.clamp(1e-6, 1 - 1e-6)
        return -(m * torch.log(m) + (1 - m) * torch.log(1 - m)).mean()

    for _ in range(int(CFG["explain_epochs"])):
        opt.zero_grad()
        masks = {
            "edge_masks":    {rel: torch.sigmoid(edge_logits[rel]) for rel in RELATIONS},
            "feature_mask":  torch.sigmoid(feat_logit),
            "temporal_mask": torch.sigmoid(temp_logit),
            "scale_mask":    torch.sigmoid(scale_logit),
        }
        out = model(x_fast, x_slow, ctx, masks=masks)
        logp_surv, logp_event = hazard_log_probs(out["hazard_logits"].float())
        surv_total = logp_surv.sum(dim=1)
        risk = 1.0 - torch.exp(surv_total)
        if target_cause is not None and target_cause >= 0:
            cum_prev = F.pad(torch.cumsum(logp_surv, dim=1), (1, 0))[:, :N_HAZARD_BINS]
            cause_mass = (torch.exp(cum_prev).unsqueeze(-1) *
                          torch.exp(logp_event)).sum(dim=1)
            target = cause_mass[:, target_cause] / cause_mass.sum(dim=1).clamp(min=1e-9)
        else:
            target = risk
        pred_loss = -torch.log(target.clamp(min=1e-9)).mean()

        reg = 0.0
        for rel in RELATIONS:
            m = torch.sigmoid(edge_logits[rel])
            reg = reg + CFG["edge_size_penalty"] * m.mean() + CFG["edge_entropy_penalty"] * entropy(m)
        fm = torch.sigmoid(feat_logit)
        reg = reg + CFG["feature_size_penalty"] * fm.mean() + CFG["feature_entropy_penalty"] * entropy(fm)
        tm = torch.sigmoid(temp_logit)
        reg = reg + CFG["temporal_size_penalty"] * tm.mean() + CFG["temporal_entropy_penalty"] * entropy(tm)
        sm = torch.sigmoid(scale_logit)
        reg = reg + CFG["scale_size_penalty"] * sm.mean()

        loss = pred_loss + reg
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = {
            "edge_masks": {rel: torch.sigmoid(edge_logits[rel]).cpu().numpy() for rel in RELATIONS},
            "feature_mask": torch.sigmoid(feat_logit).cpu().numpy(),
            "temporal_mask": torch.sigmoid(temp_logit).cpu().numpy(),
            "scale_mask": torch.sigmoid(scale_logit).cpu().numpy(),
            "base_risk": base_risk,
        }
    return final

def run_xai(model, dataset, wi, label, seed):
    """Explain up to xai_samples_per_cause near-failure windows per cause and
    write typed-edge / feature / acute-chronic / faithfulness CSVs."""
    rng = np.random.default_rng(seed)
    edge_rows, feat_rows, scale_rows, faith_rows = [], [], [], []

    # Typed expected paths per cause (ground truth from the C++ call graph).
    EXPECTED_TYPED = {
        "battery_power": {("power","battery","power_bus"), ("power","power_bus","heater_bank"),
                          ("state","battery_thermal","battery"), ("power","solar_array","power_bus")},
        "effusion_underflux": {("power","power_bus","heater_bank"), ("heat","heater_bank","effusion_cell"),
                               ("state","source_inventory","effusion_cell")},
        "effusion_undertemp": {("power","power_bus","heater_bank"), ("heat","heater_bank","effusion_cell"),
                               ("state","source_inventory","effusion_cell")},
        "substrate_undertemp": {("power","power_bus","heater_bank"), ("heat","heater_bank","substrate")},
        "stall": set(),
    }

    for cause in HAZARD_CAUSES:
        c_idx = hazard_cause_to_idx[cause]
        cand = wi[(wi["event_cause"] == c_idx) & (wi["ttf_ticks"] <= CFG["cause_eval_window_ticks"])]
        if cand.empty:
            continue
        take = cand.sample(n=min(len(cand), int(CFG["xai_samples_per_cause"])),
                           random_state=int(rng.integers(0, 2**31 - 1)))
        for local_idx in take.index.tolist():
            item = dataset[int(local_idx)]
            expl = explain_window(model, item, target_cause=c_idx)
            fast_w = float(expl["scale_mask"][0])
            slow_w = float(expl["scale_mask"][1])
            scale_rows.append({"model": label, "seed": seed, "cause": cause,
                               "window": int(local_idx),
                               "fast_weight": fast_w, "slow_weight": slow_w,
                               "chronic_ratio": slow_w / max(fast_w + slow_w, 1e-9)})
            for rel in RELATIONS:
                m = expl["edge_masks"][rel]
                flat = [(m[i, j], i, j) for i in range(N_NODES) for j in range(N_NODES) if m[i, j] > 0.05]
                flat.sort(reverse=True)
                for wgt, i, j in flat[: int(CFG["xai_top_k"])]:
                    edge_rows.append({"model": label, "seed": seed, "cause": cause,
                                      "window": int(local_idx), "relation": rel,
                                      "src": idx_to_node[i], "dst": idx_to_node[j],
                                      "weight": float(wgt)})
            fmask = expl["feature_mask"]
            for ni, node in enumerate(PHYSICAL_NODES):
                cols = MODEL_SPEC_ACTIVE.node_cols_by_node.get(node, [])
                for fi, col in enumerate(cols):
                    wgt = float(fmask[ni, fi])
                    if wgt > 0.5:
                        feat_rows.append({"model": label, "seed": seed, "cause": cause,
                                          "window": int(local_idx), "node": node,
                                          "feature": col, "weight": wgt})

        # Faithfulness: fraction of a cause's top typed edges that fall in the
        # expected typed path set (averaged over its explained windows).
        cause_edges = [r for r in edge_rows if r["cause"] == cause]
        if cause_edges and EXPECTED_TYPED.get(cause):
            top = sorted(cause_edges, key=lambda r: -r["weight"])[: int(CFG["xai_top_k"])]
            hits = sum(1 for r in top
                       if (r["relation"], r["src"], r["dst"]) in EXPECTED_TYPED[cause])
            faith_rows.append({"model": label, "seed": seed, "cause": cause,
                               "expected_path_hit_rate": hits / max(len(top), 1)})

    return (pd.DataFrame(edge_rows), pd.DataFrame(feat_rows),
            pd.DataFrame(scale_rows), pd.DataFrame(faith_rows))

# =============================================================================
# Experiment driver
# =============================================================================
MODEL_SPEC_ACTIVE = None   # set per model key inside the loop
comparison_rows = []
all_summaries = {}

for model_key in CFG["model_keys"]:
    spec = MODEL_A_SPEC if model_key == "A" else MODEL_B_SPEC
    MODEL_SPEC_ACTIVE = spec
    datasets, loaders = make_loaders(spec, CFG["batch_size"])

    for seed in CFG["seeds"]:
        label = f"model_{model_key}"
        run_label = f"{label}_s{seed}"
        log(f"===== training {run_label} ({spec.name}) =====")
        seed_everything(seed)

        model = RTGWN(
            spec=spec, n_causes=len(HAZARD_CAUSES),
            hidden_channels=CFG["hidden_channels"],
            skip_channels=CFG["skip_channels"],
            end_channels=CFG["end_channels"],
            kernel_size=CFG["kernel_size"],
            fast_layers=CFG["fast_layers"],
            slow_layers=CFG["slow_layers"],
            dropout=CFG["dropout"],
            graph_order=CFG["graph_order"],
            adaptive_adj=CFG["adaptive_adj"],
            node_embedding_dim=CFG["node_embedding_dim"],
            context_dim=len(spec.context_cols),
            context_dropout=CFG["context_dropout"],
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"{run_label}: parameters={n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                                      weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5)
        scaler = make_scaler()

        best_state, best_score, patience_left, history = None, -float("inf"), CFG["patience"], []
        for epoch in range(1, CFG["max_epochs"] + 1):
            t0 = time.perf_counter()
            tr = train_one_epoch(model, loaders["train"], optimizer, scaler, run_label, epoch)
            val_pred = predict_loader(model, loaders["val"])
            vs = summarize(val_pred, "val", run_label)
            score = validation_score(vs)
            scheduler.step(score if np.isfinite(score) else 0.0)
            lr = float(optimizer.param_groups[0]["lr"])
            history.append({
                "epoch": epoch, "train_loss": tr["loss"],
                "train_hazard_nll": tr["hazard_nll"], "train_aux_ttf": tr["aux_ttf"],
                "val_score": score, "val_auroc": vs["auroc"], "val_f1": vs["f1"],
                "val_horizon_macro_f1": vs["horizon_macro_f1"],
                "val_ttf_bin_macro_f1": vs["ttf_bin_macro_f1"],
                "val_cause_macro_f1_near": vs["cause_macro_f1_near"],
                "val_ttf_mae_ticks": vs["ttf_mae_ticks"],
                "val_ttf_mae_ticks_near": vs["ttf_mae_ticks_near"],
                "val_ece": vs["ece"],
                "learning_rate": lr, "patience_left": patience_left,
            })
            log(f"{run_label} epoch {epoch:03d} | loss={tr['loss']:.4f} "
                f"score={score:.4f} auroc={vs['auroc']:.4f} "
                f"hz={vs['horizon_macro_f1']:.3f} bin={vs['ttf_bin_macro_f1']:.3f} "
                f"cause={vs['cause_macro_f1_near']:.3f} "
                f"ttf={vs['ttf_mae_ticks']:.1f}/{vs['ttf_mae_ticks_near']:.1f} "
                f"ece={vs['ece']:.3f} lr={lr:.2e} "
                f"({time.perf_counter()-t0:.1f}s)")
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = CFG["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log(f"{run_label}: early stop at epoch {epoch}")
                    break

        pd.DataFrame(history).to_csv(OUTPUT_ROOT / f"training_history_{label}_{seed}.csv", index=False)
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save({"state_dict": model.state_dict(), "cfg": CFG,
                    "spec_name": spec.name, "cause_names": cause_names,
                    "hazard_causes": HAZARD_CAUSES,
                    "physical_nodes": PHYSICAL_NODES,
                    "typed_edges": TYPED_EDGES},
                   OUTPUT_ROOT / f"model_{model_key}_s{seed}.pt")

        # Test evaluation.
        test_pred = predict_loader(model, loaders["test"])
        ts = summarize(test_pred, "test", run_label)
        ts["val_score_best"] = best_score
        ts["n_params"] = n_params
        ts["seed"] = seed
        ts["model_key"] = model_key
        all_summaries[run_label] = {k: v for k, v in ts.items() if k != "_calibration_rows"}
        comparison_rows.append(all_summaries[run_label])

        pd.DataFrame(ts["_calibration_rows"]).assign(model=label, seed=seed).to_csv(
            OUTPUT_ROOT / "hazard_calibration.csv",
            mode="a", index=False,
            header=not (OUTPUT_ROOT / "hazard_calibration.csv").exists())

        # Per-horizon / bin / cause / regression detail CSVs.
        hor_rows = [{"model": label, "seed": seed, "horizon": int(h),
                     "f1": ts[f"horizon_{int(h)}_f1"]} for h in HORIZON_TICKS]
        pd.DataFrame(hor_rows).to_csv(OUTPUT_ROOT / "per_horizon_metrics.csv", mode="a",
                                      index=False,
                                      header=not (OUTPUT_ROOT / "per_horizon_metrics.csv").exists())
        pd.DataFrame([{"model": label, "seed": seed,
                       "ttf_bin_acc": ts["ttf_bin_acc"],
                       "ttf_bin_macro_f1": ts["ttf_bin_macro_f1"]}]).to_csv(
            OUTPUT_ROOT / "ttf_bin_metrics.csv", mode="a", index=False,
            header=not (OUTPUT_ROOT / "ttf_bin_metrics.csv").exists())
        pd.DataFrame([{"model": label, "seed": seed,
                       "ttf_mae_ticks": ts["ttf_mae_ticks"],
                       "ttf_mae_ticks_near": ts["ttf_mae_ticks_near"]}]).to_csv(
            OUTPUT_ROOT / "ttf_regression_metrics.csv", mode="a", index=False,
            header=not (OUTPUT_ROOT / "ttf_regression_metrics.csv").exists())

        near = test_pred[(test_pred["event_bin"] >= 0) &
                         (test_pred["ttf_ticks"] <= CFG["cause_eval_window_ticks"])]
        if len(near) and HAZARD_CAUSES:
            post_cols = [f"cause_post_{c}" for c in HAZARD_CAUSES]
            cp = near[post_cols].to_numpy(dtype=float).argmax(axis=1)
            ct = near["event_cause"].to_numpy(dtype=int)
            pr, rc, f1v, sup = precision_recall_fscore_support(
                ct, cp, labels=list(range(len(HAZARD_CAUSES))), zero_division=0)
            rows = [{"model": label, "seed": seed, "cause": HAZARD_CAUSES[i],
                     "precision": float(pr[i]), "recall": float(rc[i]),
                     "f1": float(f1v[i]), "support": int(sup[i])}
                    for i in range(len(HAZARD_CAUSES))]
            pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "per_class_cause_metrics.csv", mode="a",
                                      index=False,
                                      header=not (OUTPUT_ROOT / "per_class_cause_metrics.csv").exists())

        # Sample survival curves for the notebook (up to 40 test windows).
        samp = test_pred.sample(n=min(40, len(test_pred)), random_state=seed)
        surv_rows = []
        for _, r in samp.iterrows():
            for k, h in enumerate(HORIZON_TICKS):
                surv_rows.append({"model": label, "seed": seed,
                                  "window_id": int(r["window_id"]),
                                  "y_fail": int(r["y_fail"]),
                                  "ttf_ticks": float(r["ttf_ticks"]),
                                  "horizon": int(h),
                                  "survival": 1.0 - float(r[f"cdf_{int(h)}"])})
        pd.DataFrame(surv_rows).to_csv(OUTPUT_ROOT / "survival_curves_sample.csv", mode="a",
                                       index=False,
                                       header=not (OUTPUT_ROOT / "survival_curves_sample.csv").exists())

        # Learned adaptive adjacency per relation.
        for rel, mat in model.adaptive_adjacency().items():
            adj_df = pd.DataFrame(mat, index=PHYSICAL_NODES, columns=PHYSICAL_NODES)
            adj_df.to_csv(OUTPUT_ROOT / f"adaptive_adjacency_{rel}_{model_key}_{seed}.csv")

        # XAI on the primary seed only (cost control).
        if CFG["run_xai"] and seed == CFG["seeds"][0]:
            log(f"{run_label}: running typed-mask explainer ...")
            wi_test = window_index_df[window_index_df["split"].eq("test")].reset_index(drop=True)
            e_df, f_df, s_df, fa_df = run_xai(model, datasets["test"], wi_test, label, seed)
            for name, d_ in [("xai_relation_edge_masks.csv", e_df),
                             ("xai_feature_masks.csv", f_df),
                             ("xai_acute_chronic.csv", s_df),
                             ("xai_faithfulness_v12.csv", fa_df)]:
                if len(d_):
                    d_.to_csv(OUTPUT_ROOT / name, mode="a", index=False,
                              header=not (OUTPUT_ROOT / name).exists())

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del datasets, loaders
    gc.collect()

# =============================================================================
# Comparison table + run summary + sync
# =============================================================================
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
log("model comparison:\n" + comparison_df.to_string())

summary = {
    "run_tag": CFG["run_tag"],
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE),
    "cfg": {k: v for k, v in CFG.items()},
    "physical_nodes": PHYSICAL_NODES,
    "relations": {r: TYPED_EDGES[r] for r in RELATIONS},
    "hazard_causes": HAZARD_CAUSES,
    "ttf_bin_edges": TTF_BIN_EDGES,
    "n_runs": int(len(run_meta)),
    "n_configs": int(run_meta["config_key"].nunique()),
    "n_windows": int(len(window_index_df)),
    "results": all_summaries,
}
with open(OUTPUT_ROOT / "run_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
log("run_summary.json written")

with open(OUTPUT_ROOT / "train_v12_console_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(_CONSOLE_LOG_LINES))

def sync_to_drive(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)

sync_to_drive(OUTPUT_ROOT, DRIVE_OUTPUT)
log(f"done. outputs synced to {DRIVE_OUTPUT}")
