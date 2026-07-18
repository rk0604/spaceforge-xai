# feature_engineering_v2.py
# auto-converted from notebook and updated for the v11 13-node ST-GNN graph

# install missing packages when needed

import sys
import subprocess
import re

def ensure_package(pkg_name: str) -> None:
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

ensure_package("pdfplumber")
ensure_package("pyarrow")

# drive setup and user inputs

from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/gdrive")
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# set these paths before running the full pipeline

SPACEFORGE_ROOT = Path("/content/gdrive/MyDrive/SpaceForgeData")
TRACKER_PATH = SPACEFORGE_ROOT / "SpaceForge-xai Job tracker - Analytics.csv"
JOB_FILES_DIR = SPACEFORGE_ROOT / "job_files"

# Raw simulator output lives one level below this root in folders whose names
# begin with Config/config followed by a numeric id. Engineered v11 data is
# isolated in sf-cleaned-2 so it cannot collide with the superseded datasets.
RAW_DATA_ROOT = SPACEFORGE_ROOT / "spaceforge-cleaned2"
OUTPUT_ROOT = RAW_DATA_ROOT / "sf-cleaned-2"

# optional manifest file
# if this exists it should contain one row per run folder
# expected columns can include config_id run_dir job_file
MANIFEST_PATH = None

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("tracker path", TRACKER_PATH)
print("job files dir", JOB_FILES_DIR)
print("raw data root", RAW_DATA_ROOT)
print("output root", OUTPUT_ROOT)
print("in colab", IN_COLAB)

# imports constants and small helpers

import os
import re
import json
import math
import warnings
import numpy as np
import pandas as pd
import pdfplumber

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

PHASE_NAME_TO_CODE = {
    "IDLE": 0,
    "SOURCE_DEGAS": 1,
    "OXIDE_DESORB": 2,
    "SOAK": 3,
    "NUCLEATE": 4,
    "GROWTH": 5,
    "ANNEAL": 6,
    "COOLDOWN": 7,
}
PHASE_CODE_TO_NAME = {v: k for k, v in PHASE_NAME_TO_CODE.items()}

CONFIG_VALUE_COLUMNS = [
    "battery_capacity_wh",
    "battery_start_charge_wh",
    "battery_max_discharge_w",
    "battery_max_charge_w",
    "effusion_h_wk",
    "effusion_c_j",
    "solar_base_input_w",
    "solar_efficiency",
    "substrate_c_j",
    "substrate_eps",
    "substrate_fail_limit_ticks",
    "ready_band_k",
    "substrate_max_power_draw_w",
    "effusion_underflux_streak_cap",
    "effusion_undertemp_streak_cap",
    "effusion_min_flux_fraction",
    "effusion_temp_tolerance_fraction",
    "heater_bank_max_draw_w",
]

EXPECTED_FILES = {
    "array_gimbal": "ArrayGimbal.csv",
    "battery": "Battery.csv",
    "battery_thermal": "BatteryThermal.csv",
    "cryo_panel": "CryoPanel.csv",
    "effusion": "EffusionCell.csv",
    "heater_bank": "HeaterBank.csv",
    "orbit": "Orbit.csv",
    "power_bus": "PowerBus.csv",
    "process_state": "ProcessState.csv",
    "radiator": "Radiator.csv",
    "schedule_state": "ScheduleState.csv",
    "simulation_engine": "SimulationEngine.csv",
    "solar_array": "SolarArray.csv",
    "source_inventory": "SourceInventory.csv",
    "substrate": "substrate.csv",
}

MIN_REQUIRED_FILES = {
    "ArrayGimbal.csv",
    "Battery.csv",
    "BatteryThermal.csv",
    "CryoPanel.csv",
    "EffusionCell.csv",
    "HeaterBank.csv",
    "Orbit.csv",
    "PowerBus.csv",
    "ProcessState.csv",
    "Radiator.csv",
    "SimulationEngine.csv",
    "SolarArray.csv",
    "SourceInventory.csv",
    "substrate.csv",
}

NEW_GRAPH_SUBSYSTEMS = (
    "array_gimbal",
    "battery_thermal",
    "cryo_panel",
    "radiator",
    "source_inventory",
)

GRAPH_NODE_SUBSYSTEMS = (
    "orbit",
    "solar_array",
    "array_gimbal",
    "battery",
    "battery_thermal",
    "power_bus",
    "heater_bank",
    "effusion",
    "source_inventory",
    "substrate",
    "cryo_panel",
    "radiator",
    "simulation_engine",
)

# Accept Config248, config248, Config_248, Config-248, and Config 248. The
# expression is case-insensitive, but the original folder name is preserved.
CONFIG_FOLDER_PATTERN = re.compile(r"^config(?:\s*[-_]?\s*)\d+", flags=re.I)

def extract_first_int(text: str):
    """
    get the first integer from a string

    input
    text string that may contain a config id

    output
    integer or none
    """
    if text is None:
        return None
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else None

def extract_number_tokens(text: str):
    """
    extract numeric tokens from text

    input
    text raw text

    output
    list of floats
    """
    pattern = r"[-+]?\d*\.\d+(?:e[-+]?\d+)?|[-+]?\d+(?:e[-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, text, flags=re.I)]

def clean_path_string(path_obj: Path) -> str:
    """
    convert a path to a stable string

    input
    path_obj path

    output
    clean string
    """
    return str(path_obj).replace("\\", "/")

def parse_config_id_from_path(path_obj: Path):
    """
    find config id from a folder path

    input
    path_obj run folder path

    output
    config id or none
    """
    for part in [path_obj.name] + [p.name for p in path_obj.parents]:
        match = re.search(r"config\s*[_ ]*(\d+)", part, flags=re.I)
        if match:
            return int(match.group(1))
        match = re.search(r"config(\d+)", part, flags=re.I)
        if match:
            return int(match.group(1))
    return None

def consecutive_count_from_flag(flag_series: pd.Series) -> pd.Series:
    """
    compute consecutive streak counts from a binary flag

    input
    flag_series series of zero one values

    output
    series of running streak lengths
    """
    arr = flag_series.fillna(0).astype(int).to_numpy()
    out = np.zeros(len(arr), dtype=int)
    run = 0
    for i, val in enumerate(arr):
        if val == 1:
            run += 1
        else:
            run = 0
        out[i] = run
    return pd.Series(out, index=flag_series.index)

def safe_divide(numerator, denominator):
    """
    divide safely and return nan when denominator is zero

    input
    numerator array like values
    denominator array like values

    output
    array of ratios
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out

# tracker parsing

TRACKER_PATH = SPACEFORGE_ROOT / "SpaceForge-xai Job tracker - Analytics.csv"

def normalize_tracker_column_name(col_name: str) -> str:
    return " ".join(str(col_name).replace("\n", " ").split())

def normalize_config_label(label) -> str:
    text = str(label).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def read_tracker_table(tracker_path: Path) -> pd.DataFrame:
    # make sure the runtime uses the csv path
    tracker_path = Path(tracker_path)

    # read tracker file
    suffix = tracker_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(tracker_path)
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(tracker_path)
    else:
        raise ValueError(f"unsupported tracker file type {suffix}")

    # normalize columns
    raw.columns = [normalize_tracker_column_name(c) for c in raw.columns]

    # drop fully empty rows
    raw = raw.dropna(how="all").copy()

    # find config column
    possible_config_cols = ["Config", "Config #", "config", "config_id"]
    config_col = None
    for col in possible_config_cols:
        if col in raw.columns:
            config_col = col
            break

    if config_col is None:
        raise ValueError("could not find the config column in the tracker file")

    # rename tracker columns to canonical names
    column_aliases = {
        "Battery capacity_wh (battery.hpp line 9)": "battery_capacity_wh",
        "battery start capacity: charge_ (can't be set, is by default half of capacity)": "battery_start_charge_wh",
        "battery_max_discharge_W (battery.hpp line 40)": "battery_max_discharge_w",
        "battery_max_charge_W (battery.hpp line 40)": "battery_max_charge_w",
        "h_WK (effusion Cell) (too complicated leave it for Rishab)": "effusion_h_wk",
        "C_J (effusion cell) (too complicated leave it for Rishab)": "effusion_c_j",
        "base_input (solar) (main. cpp line 633)": "solar_base_input_w",
        "base_input (solar) (main.cpp line 633)": "solar_base_input_w",
        "efficiency (solar) (main. cpp line 632)": "solar_efficiency",
        "efficiency (solar) (main.cpp line 632)": "solar_efficiency",
        "C_J (susbtrateHeater.hpp line 413)": "substrate_c_j",
        "eps (substrateHeater.hpp line 411)": "substrate_eps",
        "FAIL_LIMIT_TICKS_ Subsrate (subsrateHeater.hpp line 451)": "substrate_fail_limit_ticks",
        "READY_BAND_K_ (subsrateHeater.hpp line 450)": "ready_band_k",
        "Substrate maxPowerDraw (SubsrateHeater.hpp line 115)": "substrate_max_power_draw_w",
        "Effusion Underflux streak cap (line 870 main.cpp)": "effusion_underflux_streak_cap",
        "Effusion undertemp streak cap (line 870 main.cpp)": "effusion_undertemp_streak_cap",
        "Effusion MIN_FLUX_FRACTION (line 870 main.cpp)": "effusion_min_flux_fraction",
        "Effusion TEMP_TOLERANCE_FRACTION (line 870 main.cpp)": "effusion_temp_tolerance_fraction",
        "max draw (heater bank) main.cpp line 640": "heater_bank_max_draw_w",
        "max draw (heater bank) main.cpp line 640 ": "heater_bank_max_draw_w",
    }

    rename_map = {config_col: "config_label"}
    for raw_col, canonical_col in column_aliases.items():
        if raw_col in raw.columns:
            rename_map[raw_col] = canonical_col

    tracker_df = raw.rename(columns=rename_map).copy()

    # add any missing canonical value columns
    for col in CONFIG_VALUE_COLUMNS:
        if col not in tracker_df.columns:
            tracker_df[col] = np.nan

    # keep original config names
    tracker_df["config_label"] = tracker_df["config_label"].map(normalize_config_label)

    # keep only rows that actually have a config label
    tracker_df = tracker_df[tracker_df["config_label"].notna()].copy()
    tracker_df = tracker_df[tracker_df["config_label"].astype(str).str.len() > 0].copy()

    # drop repeated header rows if they appear inside the csv
    tracker_df = tracker_df[tracker_df["config_label"].str.lower() != "config"].copy()

    # coerce numeric value columns
    for col in CONFIG_VALUE_COLUMNS:
        tracker_df[col] = pd.to_numeric(tracker_df[col], errors="coerce")

    # keep one row per original config label
    tracker_df = tracker_df[["config_label"] + CONFIG_VALUE_COLUMNS].drop_duplicates(subset=["config_label"]).reset_index(drop=True)

    # build numeric id for later merges while preserving original names
    tracker_df["config_numeric_id"] = tracker_df["config_label"].astype(str).str.extract(r"(\d+)")[0]
    tracker_df["config_numeric_id"] = pd.to_numeric(tracker_df["config_numeric_id"], errors="coerce")

    # use config 1 as fallback for missing values
    default_mask = tracker_df["config_label"].str.lower().eq("config 1")
    if not default_mask.any():
        raise ValueError("config 1 was not found in the tracker file")

    default_row = tracker_df.loc[default_mask, CONFIG_VALUE_COLUMNS].iloc[0]

    for col in CONFIG_VALUE_COLUMNS:
        tracker_df[col] = tracker_df[col].fillna(default_row[col])

    # sort while keeping original labels
    tracker_df["config_sort_group"] = tracker_df["config_label"].str.extract(r"^([A-Za-z]+)")[0].fillna("")
    tracker_df = tracker_df.sort_values(
        ["config_sort_group", "config_numeric_id", "config_label"],
        na_position="last"
    ).reset_index(drop=True)
    tracker_df = tracker_df.drop(columns=["config_sort_group"])

    return tracker_df

def build_numeric_tracker_view(tracker_df: pd.DataFrame) -> pd.DataFrame:
    # build numeric tracker view for later config folder merges
    numeric_df = tracker_df[tracker_df["config_label"].str.lower().str.startswith("config ")].copy()

    if numeric_df.empty:
        raise ValueError("no numeric config rows starting with config were found")

    numeric_df = numeric_df.rename(columns={"config_numeric_id": "config_id"})
    numeric_df = numeric_df[["config_id"] + CONFIG_VALUE_COLUMNS].copy()
    numeric_df["config_id"] = numeric_df["config_id"].astype(int)
    numeric_df = numeric_df.groupby("config_id", as_index=False).median(numeric_only=True)

    default_row = numeric_df.loc[numeric_df["config_id"] == 1, CONFIG_VALUE_COLUMNS].iloc[0]
    for col in CONFIG_VALUE_COLUMNS:
        numeric_df[col] = numeric_df[col].fillna(default_row[col])

    numeric_df = numeric_df.sort_values("config_id").reset_index(drop=True)
    return numeric_df

tracker_df = read_tracker_table(TRACKER_PATH)
tracker_numeric_df = build_numeric_tracker_view(tracker_df)

print("tracker path", TRACKER_PATH)
print("tracker all shape", tracker_df.shape)
print("tracker numeric shape", tracker_numeric_df.shape)

display(tracker_df)
display(tracker_numeric_df)

# job file parsing and per tick recipe expansion

def parse_job_file(job_file_path: Path) -> pd.DataFrame:
    """
    parse one job txt file into ordered subjob rows

    input
    job_file_path path to one job txt file

    output
    dataframe with one row per txt row

    rule logic
    the row order is preserved as subjob index
    each row defines the expected phase window and target context
    """
    rows = []

    with open(job_file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8:
                continue

            row = {
                "subjob_index": len(rows),
                "start_tick": int(float(parts[0])),
                "end_tick": int(float(parts[1])),
                "expected_flux_cm2s": float(parts[2]),
                "expected_effusion_heater_cap_w": float(parts[3]),
                "expected_mbe_on": int(parts[4]),
                "expected_substrate_on": int(parts[5]),
                "phase_name": parts[6].strip(),
                "expected_substrate_target_k": float(parts[7]),
            }

            rows.append(row)

    job_df = pd.DataFrame(rows)

    if job_df.empty:
        raise ValueError(f"no subjob rows found in {job_file_path}")

    job_df["phase_code_expected"] = job_df["phase_name"].map(PHASE_NAME_TO_CODE).astype("Int64")
    job_df["is_growth_like"] = ((job_df["expected_mbe_on"] == 1) & (job_df["expected_flux_cm2s"] > 0)).astype(int)
    job_df["is_beam_off_timed"] = ((job_df["expected_mbe_on"] == 0) & (job_df["expected_substrate_on"] == 1)).astype(int)
    job_df["is_source_only"] = ((job_df["phase_name"] == "SOURCE_DEGAS") | ((job_df["expected_mbe_on"] == 0) & (job_df["expected_substrate_on"] == 0) & (job_df["expected_effusion_heater_cap_w"] > 0))).astype(int)
    job_df["is_idle_phase"] = (job_df["phase_name"] == "IDLE").astype(int)
    job_df["expected_phase_duration_ticks"] = job_df["end_tick"] - job_df["start_tick"]

    return job_df

def expand_job_plan_to_ticks(job_df: pd.DataFrame) -> pd.DataFrame:
    """
    expand a job txt plan to one row per canonical tick

    input
    job_df parsed txt rows

    output
    dataframe with one row per canonical tick in the planned recipe

    rule logic
    each txt row is active on start tick inclusive and end tick exclusive
    """
    pieces = []

    for _, row in job_df.iterrows():
        start_tick = int(row["start_tick"])
        end_tick = int(row["end_tick"])

        if end_tick <= start_tick:
            continue

        tick_df = pd.DataFrame({"canonical_tick": np.arange(start_tick, end_tick, dtype=int)})

        for col in job_df.columns:
            tick_df[col] = row[col]

        tick_df["ticks_into_subjob"] = tick_df["canonical_tick"] - start_tick
        tick_df["ticks_remaining_in_subjob"] = end_tick - tick_df["canonical_tick"]

        pieces.append(tick_df)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)

job_lookup = {}

for path in sorted(JOB_FILES_DIR.glob("*.txt")):
    job_lookup[path.stem.lower()] = path

print("job files found", len(job_lookup))
for name in list(job_lookup.keys())[:5]:
    print(name, job_lookup[name])

sample_job_df = parse_job_file(next(iter(job_lookup.values())))
display(sample_job_df.head())

# discover run folders

# keep original config names as the main config id
tracker_df = tracker_df.copy()
if "config_row_id" not in tracker_df.columns:
    tracker_df["config_row_id"] = np.arange(1, len(tracker_df) + 1, dtype=int)

# normalize job names so folder names and txt names line up
def normalize_job_key(name: str) -> str:
    return str(name).replace(".txt", "").strip().lower()

# normalize config labels so folder names and tracker labels can be compared
def normalize_config_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())

# get the top level config folder above each run folder
def get_top_level_folder_name(run_dir: Path, raw_data_root: Path) -> str:
    rel_parts = run_dir.relative_to(raw_data_root).parts
    if len(rel_parts) == 0:
        return raw_data_root.name
    return rel_parts[0]

# build conservative folder variants without mapping different config families together
# this preserves the original config label and only strips common wrappers like data or data_xai
def build_folder_match_keys(folder_name: str):
    raw_key = normalize_config_key(folder_name)

    ordered_keys = []

    def add_key(key: str):
        if key and key not in ordered_keys:
            ordered_keys.append(key)

    add_key(raw_key)

    prefixes = ["dataxai", "data"]
    suffixes = ["data", "runs", "run"]

    current_keys = [raw_key]

    for key in list(current_keys):
        for prefix in prefixes:
            if key.startswith(prefix) and len(key) > len(prefix):
                stripped = key[len(prefix):]
                add_key(stripped)
                current_keys.append(stripped)

    for key in list(current_keys):
        for suffix in suffixes:
            if key.endswith(suffix) and len(key) > len(suffix):
                stripped = key[:-len(suffix)]
                add_key(stripped)

    return ordered_keys

# build a lookup table using the original tracker config label only
def build_tracker_label_lookup(tracker_df: pd.DataFrame):
    lookup_df = tracker_df.copy()

    lookup_df["config_id"] = lookup_df["config_label"].astype(str)
    lookup_df["config_label"] = lookup_df["config_label"].astype(str)
    lookup_df["exact_key"] = lookup_df["config_label"].apply(normalize_config_key)

    lookup_df = lookup_df[
        [
            "config_id",
            "config_label",
            "config_numeric_id",
            "config_row_id",
            "exact_key",
        ]
    ].drop_duplicates(subset=["config_label", "exact_key"]).reset_index(drop=True)

    return lookup_df

# match one folder name to one tracker row using the original config label
# first try exact normalized label match
# then try conservative wrapper stripped variants
def match_tracker_row_for_folder(folder_name: str, tracker_lookup_df: pd.DataFrame):
    folder_keys = build_folder_match_keys(folder_name)

    primary_key = folder_keys[0]
    exact_df = tracker_lookup_df.loc[tracker_lookup_df["exact_key"] == primary_key].copy()
    exact_df = exact_df.drop_duplicates(subset=["config_label"]).reset_index(drop=True)

    if len(exact_df) == 1:
        return exact_df.iloc[0].to_dict(), "folder_exact"

    if len(exact_df) > 1:
        return None, "folder_exact_ambiguous"

    for relaxed_key in folder_keys[1:]:
        relaxed_df = tracker_lookup_df.loc[tracker_lookup_df["exact_key"] == relaxed_key].copy()
        relaxed_df = relaxed_df.drop_duplicates(subset=["config_label"]).reset_index(drop=True)

        if len(relaxed_df) == 1:
            return relaxed_df.iloc[0].to_dict(), "folder_relaxed"

        if len(relaxed_df) > 1:
            return None, "folder_relaxed_ambiguous"

    return None, "folder_no_match"

# Find only top-level raw config folders. This prevents recursive discovery
# from entering sf-cleaned-2 or any unrelated directories under the raw root.
def discover_config_dirs(raw_data_root: Path):
    raw_data_root = Path(raw_data_root)
    if not raw_data_root.exists():
        raise FileNotFoundError(f"raw data root does not exist: {raw_data_root}")

    config_dirs = sorted(
        path
        for path in raw_data_root.iterdir()
        if path.is_dir() and CONFIG_FOLDER_PATTERN.match(path.name)
    )

    if not config_dirs:
        raise ValueError(
            "no top-level config folders were found under "
            f"{raw_data_root}; expected names beginning with Config/config "
            "followed by a numeric id"
        )

    return config_dirs


# find run folders by required csv files
def discover_candidate_run_dirs(raw_data_root: Path):
    candidate_dirs = []

    for config_dir in discover_config_dirs(raw_data_root):
        folders_to_check = [config_dir]
        folders_to_check.extend(path for path in config_dir.rglob("*") if path.is_dir())

        for folder in folders_to_check:
            child_names = {p.name for p in folder.iterdir() if p.is_file()}
            if MIN_REQUIRED_FILES.issubset(child_names):
                candidate_dirs.append(folder)

    if not candidate_dirs:
        required_names = ", ".join(sorted(MIN_REQUIRED_FILES))
        raise ValueError(
            "config folders were found, but none contained a complete v11 run; "
            f"required CSV files: {required_names}"
        )

    return sorted(set(candidate_dirs))

# discover all run folders and keep only configs that have all expected jobs
def discover_run_folders(raw_data_root: Path, job_lookup: dict, tracker_df: pd.DataFrame, manifest_path=None):
    tracker_lookup_df = build_tracker_label_lookup(tracker_df)

    expected_job_names = sorted([Path(p).stem for p in job_lookup.values()])
    expected_job_keys = sorted([normalize_job_key(name) for name in expected_job_names])
    expected_job_count = len(expected_job_keys)

    candidate_dirs = discover_candidate_run_dirs(raw_data_root)

    raw_rows = []

    for run_dir in candidate_dirs:
        config_folder_name = get_top_level_folder_name(run_dir, raw_data_root)
        tracker_match, match_source = match_tracker_row_for_folder(config_folder_name, tracker_lookup_df)

        job_name_raw = run_dir.name.strip()
        job_key = normalize_job_key(job_name_raw)
        job_path = job_lookup.get(job_key)

        raw_rows.append(
            {
                "config_folder_name": config_folder_name,
                "tracker_match_found": tracker_match is not None,
                "config_id": str(tracker_match["config_id"]) if tracker_match is not None else np.nan,
                "config_label": str(tracker_match["config_label"]) if tracker_match is not None else np.nan,
                "config_numeric_id": tracker_match["config_numeric_id"] if tracker_match is not None else np.nan,
                "config_row_id": int(tracker_match["config_row_id"]) if tracker_match is not None else np.nan,
                "config_inference_source": match_source,
                "job_name": Path(job_path).stem if job_path is not None else job_name_raw,
                "job_key": job_key,
                "job_match_found": job_path is not None,
                "job_file_path": clean_path_string(Path(job_path)) if job_path is not None else np.nan,
                "run_dir": clean_path_string(run_dir),
            }
        )

    raw_manifest = pd.DataFrame(raw_rows)

    if raw_manifest.empty:
        raise ValueError("no run folders were discovered")

    # summarize each config folder before filtering
    summary_rows = []

    for config_folder_name, group in raw_manifest.groupby("config_folder_name", dropna=False):
        matched_job_keys = sorted(group.loc[group["job_match_found"] == True, "job_key"].dropna().unique().tolist())
        missing_job_keys = sorted(set(expected_job_keys) - set(matched_job_keys))

        tracker_match_found = bool(group["tracker_match_found"].any())

        reason_parts = []

        if not tracker_match_found:
            reason_parts.append("config label not matched")

        if len(matched_job_keys) < expected_job_count:
            reason_parts.append(f"missing {expected_job_count - len(matched_job_keys)} jobs")

        if len(matched_job_keys) > expected_job_count:
            reason_parts.append("extra jobs found")

        include_config = tracker_match_found and len(matched_job_keys) == expected_job_count

        first_row = group.iloc[0]

        summary_rows.append(
            {
                "config_folder_name": config_folder_name,
                "config_id": first_row["config_id"] if tracker_match_found else np.nan,
                "config_label": first_row["config_label"] if tracker_match_found else np.nan,
                "config_numeric_id": first_row["config_numeric_id"] if tracker_match_found else np.nan,
                "config_row_id": first_row["config_row_id"] if tracker_match_found else np.nan,
                "config_inference_source": first_row["config_inference_source"],
                "matched_job_count": len(matched_job_keys),
                "expected_job_count": expected_job_count,
                "missing_job_names": ", ".join(missing_job_keys),
                "include_config": include_config,
                "exclude_reason": "complete" if include_config else " and ".join(reason_parts),
            }
        )

    config_summary_df = pd.DataFrame(summary_rows).sort_values("config_folder_name").reset_index(drop=True)

    # keep only configs that have all expected jobs and a matched config label
    keep_folders = set(config_summary_df.loc[config_summary_df["include_config"] == True, "config_folder_name"].tolist())

    run_manifest = raw_manifest.loc[raw_manifest["config_folder_name"].isin(keep_folders)].copy()
    run_manifest = run_manifest.loc[run_manifest["job_match_found"] == True].copy()

    run_manifest = run_manifest[
        [
            "config_id",
            "config_label",
            "config_numeric_id",
            "config_row_id",
            "config_folder_name",
            "config_inference_source",
            "job_name",
            "job_file_path",
            "run_dir",
        ]
    ].copy()

    # build a stable run id using the original config label
    run_manifest["run_id"] = run_manifest.apply(
        lambda row: f"{normalize_config_key(row['config_label'])}__{row['job_name']}",
        axis=1
    )

    run_manifest = run_manifest.drop_duplicates(subset=["run_dir"]).sort_values(["config_label", "job_name"]).reset_index(drop=True)

    included_configs_df = config_summary_df.loc[config_summary_df["include_config"] == True].copy()
    excluded_configs_df = config_summary_df.loc[config_summary_df["include_config"] == False].copy()

    return run_manifest, included_configs_df, excluded_configs_df

# build the filtered run manifest
run_manifest, included_configs_df, excluded_configs_df = discover_run_folders(RAW_DATA_ROOT, job_lookup, tracker_df, MANIFEST_PATH)

# show how many runs and configs remain after filtering
print("runs discovered after filtering", len(run_manifest))
print("configs kept", len(included_configs_df))
print("configs excluded", len(excluded_configs_df))

# show the kept config folders
display(
    included_configs_df[
        [
            "config_folder_name",
            "config_label",
            "matched_job_count",
            "expected_job_count",
            "config_inference_source",
        ]
    ].sort_values(["config_label", "config_folder_name"]).reset_index(drop=True)
)

# show which configs were excluded and why
display(
    excluded_configs_df[
        [
            "config_folder_name",
            "config_label",
            "matched_job_count",
            "expected_job_count",
            "missing_job_names",
            "exclude_reason",
            "config_inference_source",
        ]
    ].sort_values(["config_folder_name"]).reset_index(drop=True)
)

# show the first kept run rows for inspection
display(run_manifest.head(50))

# raw log loading and native tick alignment
#
# this cell keeps the subsystem timeline on its native raw tick axis
# processstate and simulationengine are not shifted backward anymore
#
# important logic
# the true failure cause tick should be determined later from the raw subsystem evidence
# that means the kth bad raw tick is the true failure tick for undertemp or underflux streak failures
# processstate and simulationengine can still show the abort one tick later as a state transition artifact
# that late state log should remain visible in the data instead of forcing the full timeline backward here
#
# so this cell does five things
# one reads each subsystem csv for a run
# two optionally aligns schedulestate to processstate by agreement scoring
# three creates a canonical tick axis that stays equal to the raw tick for the core subsystem logs
# four defines merge helpers for building one per tick dataframe
# five runs a sample sanity check and builds one sample merged dataframe

def read_one_csv(path: Path) -> pd.DataFrame:
    """
    read one csv and strip column whitespace

    input
    path csv path

    output
    dataframe
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_run_logs(run_dir: Path) -> dict:
    """
    load all known subsystem csv files for one run

    input
    run_dir run folder path

    output
    dict of subsystem dataframes
    """
    logs = {}

    for key, filename in EXPECTED_FILES.items():
        csv_path = run_dir / filename
        if csv_path.exists():
            logs[key] = read_one_csv(csv_path)
        else:
            logs[key] = pd.DataFrame()

    empty_graph_logs = [
        key for key in GRAPH_NODE_SUBSYSTEMS
        if key not in logs or logs[key].empty
    ]
    if empty_graph_logs:
        raise ValueError(
            f"run {run_dir} has empty or missing graph-node logs: "
            + ", ".join(empty_graph_logs)
        )

    return logs


def choose_schedule_shift(schedule_df: pd.DataFrame, process_df: pd.DataFrame):
    """
    choose the best schedulestate shift by agreement with native processstate

    input
    schedule_df raw schedulestate
    process_df raw processstate on its native tick axis

    output
    best shift and agreement score

    rule logic
    agreement is measured on controlling job index phase code and mbe flag
    processstate is treated as the native reference timeline
    this does not relabel failure causes
    this only improves schedule table alignment when needed
    """
    if schedule_df.empty or process_df.empty:
        return 0, np.nan

    best_shift = 0
    best_score = -1.0

    compare_cols = ["controlling_job_index", "phase_code", "mbe_flag"]

    process_temp = process_df.copy()
    if "tick" not in process_temp.columns:
        raise ValueError("tick column missing in process_state")

    process_temp["canonical_tick"] = process_temp["tick"]
    process_keep = ["canonical_tick"] + [c for c in compare_cols if c in process_temp.columns]
    process_small = process_temp[process_keep].copy()

    for shift in [-2, -1, 0, 1, 2]:
        temp = schedule_df.copy()

        if "tick" not in temp.columns:
            raise ValueError("tick column missing in schedule_state")

        temp["canonical_tick"] = temp["tick"] + shift
        temp = temp[temp["canonical_tick"] >= 0].copy()

        schedule_keep = ["canonical_tick"] + [c for c in compare_cols if c in temp.columns]
        schedule_small = temp[schedule_keep].copy()

        merged = process_small.merge(schedule_small, on="canonical_tick", suffixes=("_proc", "_sched"))

        if merged.empty:
            continue

        score_parts = []

        if "controlling_job_index_proc" in merged.columns and "controlling_job_index_sched" in merged.columns:
            score_parts.append((merged["controlling_job_index_proc"] == merged["controlling_job_index_sched"]).mean())

        if "phase_code_proc" in merged.columns and "phase_code_sched" in merged.columns:
            score_parts.append((merged["phase_code_proc"] == merged["phase_code_sched"]).mean())

        if "mbe_flag_proc" in merged.columns and "mbe_flag_sched" in merged.columns:
            score_parts.append((merged["mbe_flag_proc"] == merged["mbe_flag_sched"]).mean())

        if not score_parts:
            continue

        score = float(np.mean(score_parts))

        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift, best_score


def shift_and_align_logs(logs: dict) -> dict:
    """
    prepare aligned logs on a canonical tick axis without shifting core state logs backward

    input
    logs dict of subsystem dataframes

    output
    dict of aligned dataframes

    rule logic
    most logs keep canonical tick equal to raw tick
    processstate stays on its native tick
    simulationengine stays on its native tick
    schedulestate can be shifted if needed to better match raw processstate
    negative canonical ticks are dropped only when a chosen schedule shift creates them

    important note
    this cell does not stamp the true failure cause tick
    later failure engineering should use the raw subsystem streak evidence and the config caps
    that lets the kth bad subsystem tick remain the true failure tick
    while still preserving the late abort row in simulationengine and processstate as an audit artifact
    """
    aligned = {}

    process_df_native = pd.DataFrame()

    for key, df in logs.items():
        if df.empty:
            aligned[key] = df.copy()
            continue

        temp = df.copy()

        if "tick" not in temp.columns:
            raise ValueError(f"tick column missing in {key}")

        # keep the original raw tick for traceability
        temp["raw_tick"] = temp["tick"]

        # native canonical tick
        # no one tick backward shift is applied to processstate or simulationengine
        temp["canonical_tick"] = temp["tick"]

        if key == "process_state":
            process_df_native = temp.copy()

        aligned[key] = temp

    # optionally align schedulestate against the native processstate timeline
    if "schedule_state" in logs and not logs["schedule_state"].empty:
        if not process_df_native.empty:
            shift_value, align_score = choose_schedule_shift(logs["schedule_state"], process_df_native)
        else:
            shift_value, align_score = 0, np.nan

        temp = logs["schedule_state"].copy()
        temp["raw_tick"] = temp["tick"]
        temp["canonical_tick"] = temp["tick"] + shift_value
        temp["schedule_shift_used"] = shift_value
        temp["schedule_alignment_score"] = align_score
        temp = temp[temp["canonical_tick"] >= 0].copy()
        aligned["schedule_state"] = temp

    return aligned


def prefix_merge(base_df: pd.DataFrame, subsystem_df: pd.DataFrame, subsystem_name: str) -> pd.DataFrame:
    """
    merge one subsystem dataframe into the canonical base

    input
    base_df base tick dataframe
    subsystem_df aligned subsystem dataframe
    subsystem_name prefix name

    output
    merged dataframe
    """
    if subsystem_df.empty:
        return base_df

    temp = subsystem_df.copy()

    renamed = {
        col: f"{subsystem_name}__{col}"
        for col in temp.columns
        if col != "canonical_tick"
    }
    temp = temp.rename(columns=renamed)

    return base_df.merge(temp, on="canonical_tick", how="left")


def build_canonical_base(aligned_logs: dict) -> pd.DataFrame:
    """
    build a base canonical tick dataframe across all available subsystem logs

    input
    aligned_logs dict of aligned subsystem dataframes

    output
    dataframe with one row per canonical tick

    rule logic
    the base axis is the union of all canonical ticks present across subsystems
    """
    tick_pieces = []

    for df in aligned_logs.values():
        if df.empty:
            continue
        if "canonical_tick" not in df.columns:
            continue
        tick_pieces.append(df["canonical_tick"])

    if not tick_pieces:
        return pd.DataFrame({"canonical_tick": pd.Series(dtype=int)})

    all_ticks = pd.concat(tick_pieces, ignore_index=True).dropna().astype(int)
    base_df = pd.DataFrame({"canonical_tick": np.sort(all_ticks.unique())})
    return base_df.reset_index(drop=True)


def build_run_tick_df(aligned_logs: dict) -> pd.DataFrame:
    """
    build one wide per tick dataframe for a run

    input
    aligned_logs dict of aligned subsystem dataframes

    output
    wide dataframe with prefixed subsystem columns

    rule logic
    each subsystem is merged onto one canonical tick base
    raw tick columns are preserved under each subsystem prefix for auditability
    """
    base_df = build_canonical_base(aligned_logs)

    for subsystem_name, subsystem_df in aligned_logs.items():
        base_df = prefix_merge(base_df, subsystem_df, subsystem_name)

    base_df = base_df.sort_values("canonical_tick").reset_index(drop=True)
    return base_df


# sample sanity check on one discovered run
sample_run = run_manifest.iloc[0].to_dict()
sample_logs = load_run_logs(Path(sample_run["run_dir"]))
sample_aligned_logs = shift_and_align_logs(sample_logs)
sample_tick_df = build_run_tick_df(sample_aligned_logs)

print("sample run id", sample_run["run_id"])
print("sample run dir", sample_run["run_dir"])

for key, df in sample_aligned_logs.items():
    if not df.empty:
        cols_to_show = [c for c in ["tick", "raw_tick", "canonical_tick"] if c in df.columns]
        print(key, df[cols_to_show].head(3).to_dict("records"))
        print(key, "canonical tick min", int(df["canonical_tick"].min()), "canonical tick max", int(df["canonical_tick"].max()), "shape", df.shape)

print("sample merged tick df shape", sample_tick_df.shape)
display(sample_tick_df.head(20))

# build merged per tick dataframes for all runs
#
# this cell loops through every run in run_manifest
# loads the raw subsystem logs
# keeps processstate and simulationengine on their native raw tick axis
# builds one merged per tick dataframe for each run
# attaches run level metadata and config values to every tick row
# saves one parquet file per run and one combined parquet file for all runs
#
# important logic
# this cell does not decide the true failure cause tick
# that should be done later from the raw subsystem streak evidence and the config caps
# this cell only prepares the full per tick dataset needed for that later feature engineering step

def get_tracker_context_row(run_row_dict: dict, tracker_df: pd.DataFrame) -> pd.Series:
    """
    get the tracker row for one run using the original config label

    input
    run_row_dict one row from run_manifest as a dict
    tracker_df cleaned tracker dataframe

    output
    one tracker row as a series
    """
    config_label = str(run_row_dict.get("config_label", "")).strip()

    matches = tracker_df.loc[tracker_df["config_label"].astype(str).str.strip() == config_label].copy()

    if matches.empty:
        fallback = pd.Series(index=["config_label"] + CONFIG_VALUE_COLUMNS, dtype="object")
        fallback["config_label"] = config_label
        for col in CONFIG_VALUE_COLUMNS:
            fallback[col] = np.nan
        return fallback

    return matches.iloc[0]


def safe_filename(text: str) -> str:
    """
    convert text to a filesystem safe file name

    input
    text raw name

    output
    safe string
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


PER_RUN_OUTPUT_DIR = OUTPUT_ROOT / "per_run_tick_frames"
PER_RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_run_tick_dfs = []
build_summary_rows = []
build_error_rows = []

print("total runs to build", len(run_manifest))

for run_idx, (_, run_row) in enumerate(run_manifest.iterrows(), start=1):
    run_row_dict = run_row.to_dict()
    run_id = str(run_row_dict["run_id"])
    run_dir = Path(run_row_dict["run_dir"])

    try:
        # load and align raw subsystem logs on the native tick axis
        logs = load_run_logs(run_dir)
        aligned_logs = shift_and_align_logs(logs)
        run_tick_df = build_run_tick_df(aligned_logs)

        # attach run level identifiers to every row
        run_tick_df.insert(0, "run_id", run_id)
        run_tick_df.insert(1, "config_id", run_row_dict.get("config_id"))
        run_tick_df.insert(2, "config_label", run_row_dict.get("config_label"))
        run_tick_df.insert(3, "config_numeric_id", run_row_dict.get("config_numeric_id"))
        run_tick_df.insert(4, "config_row_id", run_row_dict.get("config_row_id"))
        run_tick_df.insert(5, "config_folder_name", run_row_dict.get("config_folder_name"))
        run_tick_df.insert(6, "job_name", run_row_dict.get("job_name"))
        run_tick_df.insert(7, "job_file_path", run_row_dict.get("job_file_path"))
        run_tick_df.insert(8, "run_dir", run_row_dict.get("run_dir"))

        # attach tracker config values to every tick row
        # this keeps the failure caps and other config settings available during later labeling
        tracker_row = get_tracker_context_row(run_row_dict, tracker_df)
        for col in CONFIG_VALUE_COLUMNS:
            run_tick_df[col] = tracker_row.get(col, np.nan)

        # optionally merge the expected per tick recipe from the job txt file
        # this only runs if the parsing functions were already defined in an earlier cell
        if "parse_job_file" in globals() and "expand_job_plan_to_ticks" in globals():
            job_file_path = run_row_dict.get("job_file_path", np.nan)

            if pd.notna(job_file_path):
                parsed_job_df = parse_job_file(Path(job_file_path))
                expected_tick_df = expand_job_plan_to_ticks(parsed_job_df)

                if not expected_tick_df.empty:
                    run_tick_df = run_tick_df.merge(
                        expected_tick_df,
                        on="canonical_tick",
                        how="left"
                    )

        # sort for stable downstream work
        run_tick_df = run_tick_df.sort_values("canonical_tick").reset_index(drop=True)

        # save one parquet per run
        per_run_path = PER_RUN_OUTPUT_DIR / f"{safe_filename(run_id)}.parquet"
        run_tick_df.to_parquet(per_run_path, index=False)

        # collect summary info
        schedule_shift_used = np.nan
        if "schedule_state__schedule_shift_used" in run_tick_df.columns:
            non_null_shift = run_tick_df["schedule_state__schedule_shift_used"].dropna()
            if len(non_null_shift) > 0:
                schedule_shift_used = non_null_shift.iloc[0]

        build_summary_rows.append(
            {
                "run_id": run_id,
                "config_label": run_row_dict.get("config_label"),
                "job_name": run_row_dict.get("job_name"),
                "tick_min": run_tick_df["canonical_tick"].min() if not run_tick_df.empty else np.nan,
                "tick_max": run_tick_df["canonical_tick"].max() if not run_tick_df.empty else np.nan,
                "n_ticks": len(run_tick_df),
                "n_columns": run_tick_df.shape[1],
                "schedule_shift_used": schedule_shift_used,
                "output_path": clean_path_string(per_run_path),
            }
        )

        all_run_tick_dfs.append(run_tick_df)

        if run_idx % 25 == 0 or run_idx == len(run_manifest):
            print("built", run_idx, "of", len(run_manifest), "runs")

    except Exception as exc:
        build_error_rows.append(
            {
                "run_id": run_id,
                "config_label": run_row_dict.get("config_label"),
                "job_name": run_row_dict.get("job_name"),
                "run_dir": run_row_dict.get("run_dir"),
                "error": str(exc),
            }
        )
        print("failed", run_id, str(exc))


build_summary_df = pd.DataFrame(build_summary_rows).sort_values(
    ["config_label", "job_name"],
    na_position="last"
).reset_index(drop=True)

build_errors_df = pd.DataFrame(build_error_rows)

if all_run_tick_dfs:
    all_runs_tick_df = pd.concat(all_run_tick_dfs, ignore_index=True, sort=False)
    all_runs_tick_df = all_runs_tick_df.sort_values(
        ["config_label", "job_name", "canonical_tick"],
        na_position="last"
    ).reset_index(drop=True)
else:
    all_runs_tick_df = pd.DataFrame()

# save combined outputs
combined_parquet_path = OUTPUT_ROOT / "all_runs_tick_df.parquet"
summary_csv_path = OUTPUT_ROOT / "all_runs_build_summary.csv"
error_csv_path = OUTPUT_ROOT / "all_runs_build_errors.csv"

if not all_runs_tick_df.empty:
    all_runs_tick_df.to_parquet(combined_parquet_path, index=False)

build_summary_df.to_csv(summary_csv_path, index=False)

if not build_errors_df.empty:
    build_errors_df.to_csv(error_csv_path, index=False)

print("runs requested", len(run_manifest))
print("runs built", len(build_summary_df))
print("runs failed", len(build_errors_df))
print("combined tick df shape", all_runs_tick_df.shape)
print("per run output dir", PER_RUN_OUTPUT_DIR)
print("combined parquet path", combined_parquet_path)
print("summary csv path", summary_csv_path)
if not build_errors_df.empty:
    print("error csv path", error_csv_path)

display(build_summary_df.head(20))

if not build_errors_df.empty:
    display(build_errors_df)

display(all_runs_tick_df.head(20))

# ignore warnings of this type
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# canonical signal resolution and engineered rule features
#
# this cell sits after the all runs merged tick dataframe is built
# it standardizes raw subsystem columns into stable canonical signals
# and then builds rule based features per run without relying on any shift based logic
#
# important logic
# feature engineering must run per run
# this prevents streak counts phase ages and other sequential features from bleeding across runs
#
# this cell does not replace the raw merged dataframe
# it creates a separate engineered dataframe so the funnel can still keep using the raw merged columns if needed

# exact raw column lists by subsystem
# edit these lists when the tracked raw columns change

SUBSYSTEM_COLUMN_ARRAYS = {
    "schedule_state": [
        "tick",
        "controlling_job_index",
        "scheduler_state_code",
        "prep_mode_code",
        "requested_start_tick",
        "requested_end_tick",
        "requested_phase_duration_ticks",
        "requested_live_duration_ticks",
        "actual_queue_enter_tick",
        "actual_thermal_prep_start_tick",
        "actual_warmup_start_tick",
        "actual_cooldown_start_tick",
        "actual_phase_start_tick",
        "actual_phase_end_tick",
        "actual_deposition_start_tick",
        "actual_deposition_end_tick",
        "phase_ticks_completed",
        "remaining_phase_ticks",
        "live_ticks_completed",
        "remaining_live_ticks",
        "delay_from_requested_start",
        "phase_code",
        "mbe_on",
        "substrate_on",
        "phase_ready_for_execution",
        "queued_active",
        "warmup_active",
        "cooldown_active",
        "thermal_prep_active",
        "live_active",
        "done_active",
        "aborted_active",
        "deposition_requested",
        "mbe_flag",
    ],
    "simulation_engine": [
        "tick",
        "status",
        "bus_remaining_W",
        "solar_output_W",
        "job_failed",
        "spacecraft_base_load_W",
        "total_power_requested_W",
        "total_power_granted_W",
        "total_power_generated_W",
        "battery_capacity_Wh",
    ],
    "substrate": [
        "tick",
        "job_index",
        "job_active",
        "substrate_control_on",
        "T_sub_K",
        "T_target_K",
        "T_env_eff_K",
        "solar_scale",
        "P_solar_abs_W",
        "P_req_W",
        "P_deliv_W",
        "P_loss_W",
        "streak",
        "C_J",
        "eps",
        "h_WK",
    ],
    "effusion": [
        "tick",
        "act_temp_K",
        "target_temp_K",
        "T_env_eff_K",
        "solar_scale",
        "P_solar_abs_W",
        "heatInput_w",
        "underflux_streak",
        "temp_miss_streak",
        "P_loss_W",
        "P_net_W",
        "C_J",
        "h_WK",
    ],
    "process_state": [
        "tick",
        "controlling_job_index",
        "mbe_on",
        "substrate_on",
        "raw_job_flux_cm2s",
        "deposition_requested",
        "phase_ready_for_execution",
        "effusionDemand_W",
        "substrateDemand_W",
        "effusion_ready",
        "wafer_ready",
        "mbe_flag",
        "effusion_delivered_W",
        "substrate_delivered_W",
    ],
    "solar_array": [
        "tick",
        "solar_scale",
        "output",
    ],
    "battery": [
        "tick",
        "charge_Wh",
        "max_charge_W",
        "max_discharge_W",
    ],
    "heater_bank": [
        "tick",
        "eff_requested_W",
        "eff_delivered_W",
        "sub_requested_W",
        "sub_delivered_W",
        "priority_substrate",
    ],
    "power_bus": [
        "tick",
        "solar_added",
        "requested",
        "granted",
        "batt_drawn",
    ],
    "array_gimbal": [
        "tick", "status", "sun_angle_deg", "gimbal_angle_deg",
        "pointing_err_deg", "pointing_eff", "slew_rate_deg_s", "in_sun",
        "power_req_W", "power_granted_W",
    ],
    "battery_thermal": [
        "tick", "status", "T_batt_K", "T_sink_K", "q_joule_W", "q_cool_W",
        "batt_current_A", "surv_heater_req_W", "surv_heater_granted_W",
        "derate_discharge", "derate_charge", "eff_max_discharge_W",
        "eff_max_charge_W", "prev_discharge_W", "prev_charge_W",
    ],
    "cryo_panel": [
        "tick", "status", "mode_regen", "T_cold_K", "duty", "power_req_W",
        "power_granted_W", "q_lift_W", "q_parasitic_W", "adsorbed_g",
        "ads_rate_g_min", "regen_ticks_left", "heat_to_radiator_W",
    ],
    "radiator": [
        "tick", "status", "T_loop_K", "louver_frac", "T_sink_K", "q_load_W",
        "q_reject_W", "actuator_req_W", "actuator_granted_W",
    ],
    "source_inventory": [
        "tick", "status", "remaining_g", "remaining_frac",
        "deplete_rate_g_min", "live_deposition", "cell_temp_K",
        "c_eff_J_per_K", "temp_bias_frac",
    ],
}

# exact source map
# the first existing merged column wins
# this keeps the feature selection easy to edit

CANONICAL_SIGNAL_SOURCES = {
    "schedule_raw_tick": [
        ("schedule_state", "tick"),
    ],
    "process_raw_tick": [
        ("process_state", "tick"),
    ],
    "simulation_raw_tick": [
        ("simulation_engine", "tick"),
    ],
    "heater_raw_tick": [
        ("heater_bank", "tick"),
    ],
    "substrate_raw_tick": [
        ("substrate", "tick"),
    ],
    "effusion_raw_tick": [
        ("effusion", "tick"),
    ],
    "battery_raw_tick": [
        ("battery", "tick"),
    ],
    "power_bus_raw_tick": [
        ("power_bus", "tick"),
    ],
    "solar_array_raw_tick": [
        ("solar_array", "tick"),
    ],
    "array_gimbal_raw_tick": [
        ("array_gimbal", "tick"),
    ],
    "battery_thermal_raw_tick": [
        ("battery_thermal", "tick"),
    ],
    "cryo_panel_raw_tick": [
        ("cryo_panel", "tick"),
    ],
    "radiator_raw_tick": [
        ("radiator", "tick"),
    ],
    "source_inventory_raw_tick": [
        ("source_inventory", "tick"),
    ],
    "schedule_controlling_job_index": [
        ("schedule_state", "controlling_job_index"),
    ],
    "schedule_phase_code": [
        ("schedule_state", "phase_code"),
    ],
    "schedule_mbe_flag": [
        ("schedule_state", "mbe_flag"),
    ],
    "schedule_mbe_on": [
        ("schedule_state", "mbe_on"),
    ],
    "schedule_substrate_on": [
        ("schedule_state", "substrate_on"),
    ],
    "schedule_phase_ready_for_execution": [
        ("schedule_state", "phase_ready_for_execution"),
    ],
    "schedule_deposition_requested": [
        ("schedule_state", "deposition_requested"),
    ],
    "schedule_aborted_active": [
        ("schedule_state", "aborted_active"),
    ],
    "schedule_live_active": [
        ("schedule_state", "live_active"),
    ],
    "schedule_warmup_active": [
        ("schedule_state", "warmup_active"),
    ],
    "schedule_cooldown_active": [
        ("schedule_state", "cooldown_active"),
    ],
    "schedule_thermal_prep_active": [
        ("schedule_state", "thermal_prep_active"),
    ],
    "simulation_job_failed": [
        ("simulation_engine", "job_failed"),
    ],
    "simulation_status": [
        ("simulation_engine", "status"),
    ],
    "simulation_bus_remaining_w": [
        ("simulation_engine", "bus_remaining_W"),
    ],
    "simulation_total_power_requested_w": [
        ("simulation_engine", "total_power_requested_W"),
        ("power_bus", "requested"),
    ],
    "simulation_total_power_granted_w": [
        ("simulation_engine", "total_power_granted_W"),
        ("power_bus", "granted"),
    ],
    "simulation_total_power_generated_w": [
        ("simulation_engine", "total_power_generated_W"),
    ],
    "simulation_solar_output_w": [
        ("simulation_engine", "solar_output_W"),
        ("solar_array", "output"),
        ("power_bus", "solar_added"),
    ],
    "battery_capacity_wh_logged": [
        ("simulation_engine", "battery_capacity_Wh"),
    ],
    "substrate_job_index": [
        ("substrate", "job_index"),
    ],
    "substrate_job_active": [
        ("substrate", "job_active"),
    ],
    "substrate_control_on_logged": [
        ("substrate", "substrate_control_on"),
    ],
    "substrate_actual_temp_k": [
        ("substrate", "T_sub_K"),
    ],
    "substrate_target_temp_k_logged": [
        ("substrate", "T_target_K"),
    ],
    "substrate_env_temp_k": [
        ("substrate", "T_env_eff_K"),
    ],
    "substrate_solar_scale": [
        ("substrate", "solar_scale"),
        ("solar_array", "solar_scale"),
        ("effusion", "solar_scale"),
    ],
    "substrate_solar_absorbed_w": [
        ("substrate", "P_solar_abs_W"),
    ],
    "substrate_requested_power_w_logged": [
        ("substrate", "P_req_W"),
        ("heater_bank", "sub_requested_W"),
        ("process_state", "substrateDemand_W"),
    ],
    "substrate_delivered_power_w": [
        ("substrate", "P_deliv_W"),
        ("heater_bank", "sub_delivered_W"),
        ("process_state", "substrate_delivered_W"),
    ],
    "substrate_power_loss_w": [
        ("substrate", "P_loss_W"),
    ],
    "substrate_temp_miss_streak_logged": [
        ("substrate", "streak"),
    ],
    "substrate_c_j": [
        ("substrate", "C_J"),
    ],
    "substrate_eps": [
        ("substrate", "eps"),
    ],
    "substrate_h_wk": [
        ("substrate", "h_WK"),
    ],
    "effusion_actual_temp_k": [
        ("effusion", "act_temp_K"),
    ],
    "effusion_target_temp_k_logged": [
        ("effusion", "target_temp_K"),
    ],
    "effusion_env_temp_k": [
        ("effusion", "T_env_eff_K"),
    ],
    "effusion_solar_scale": [
        ("effusion", "solar_scale"),
        ("solar_array", "solar_scale"),
        ("substrate", "solar_scale"),
    ],
    "effusion_solar_absorbed_w": [
        ("effusion", "P_solar_abs_W"),
    ],
    "effusion_heat_input_w": [
        ("effusion", "heatInput_w"),
    ],
    "effusion_underflux_streak_logged": [
        ("effusion", "underflux_streak"),
    ],
    "effusion_temp_miss_streak_logged": [
        ("effusion", "temp_miss_streak"),
    ],
    "effusion_power_loss_w": [
        ("effusion", "P_loss_W"),
    ],
    "effusion_power_net_w": [
        ("effusion", "P_net_W"),
    ],
    "effusion_requested_power_w_logged": [
        ("heater_bank", "eff_requested_W"),
        ("process_state", "effusionDemand_W"),
        ("effusion", "heatInput_w"),
    ],
    "effusion_delivered_power_w": [
        ("heater_bank", "eff_delivered_W"),
        ("process_state", "effusion_delivered_W"),
    ],
    "effusion_c_j": [
        ("effusion", "C_J"),
    ],
    "effusion_h_wk": [
        ("effusion", "h_WK"),
    ],
    "process_controlling_job_index": [
        ("process_state", "controlling_job_index"),
    ],
    "process_mbe_on": [
        ("process_state", "mbe_on"),
    ],
    "process_substrate_on": [
        ("process_state", "substrate_on"),
    ],
    "process_raw_job_flux_cm2s": [
        ("process_state", "raw_job_flux_cm2s"),
    ],
    "process_deposition_requested": [
        ("process_state", "deposition_requested"),
    ],
    "process_phase_ready_for_execution": [
        ("process_state", "phase_ready_for_execution"),
    ],
    "process_effusion_demand_w": [
        ("process_state", "effusionDemand_W"),
    ],
    "process_substrate_demand_w": [
        ("process_state", "substrateDemand_W"),
    ],
    "process_effusion_ready": [
        ("process_state", "effusion_ready"),
    ],
    "process_wafer_ready": [
        ("process_state", "wafer_ready"),
    ],
    "process_mbe_flag": [
        ("process_state", "mbe_flag"),
    ],
    "process_effusion_delivered_w": [
        ("process_state", "effusion_delivered_W"),
    ],
    "process_substrate_delivered_w": [
        ("process_state", "substrate_delivered_W"),
    ],
    "solar_array_output_w": [
        ("solar_array", "output"),
        ("simulation_engine", "solar_output_W"),
        ("power_bus", "solar_added"),
    ],
    "battery_charge_wh_raw": [
        ("battery", "charge_Wh"),
    ],
    "battery_max_charge_w_logged": [
        ("battery", "max_charge_W"),
    ],
    "battery_max_discharge_w_logged": [
        ("battery", "max_discharge_W"),
    ],
    "battery_power_w_raw": [
        ("power_bus", "batt_drawn"),
    ],
    "power_bus_solar_added_w": [
        ("power_bus", "solar_added"),
    ],
    "power_bus_requested_w": [
        ("power_bus", "requested"),
        ("simulation_engine", "total_power_requested_W"),
    ],
    "power_bus_granted_w": [
        ("power_bus", "granted"),
        ("simulation_engine", "total_power_granted_W"),
    ],
}


def merged_col_name(subsystem_name: str, raw_col_name: str) -> str:
    return f"{subsystem_name}__{raw_col_name}"


def first_existing_exact_column(df: pd.DataFrame, source_list):
    for subsystem_name, raw_col_name in source_list:
        candidate_col = merged_col_name(subsystem_name, raw_col_name)
        if candidate_col in df.columns:
            return candidate_col
    return None


def numeric_or_nan(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def object_or_na(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name in df.columns:
        return df[col_name].astype("object")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def numeric_from_resolved(df: pd.DataFrame, resolved: dict, signal_name: str) -> pd.Series:
    source_col = resolved.get(signal_name)
    if source_col is None or source_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[source_col], errors="coerce")


def object_from_resolved(df: pd.DataFrame, resolved: dict, signal_name: str) -> pd.Series:
    source_col = resolved.get(signal_name)
    if source_col is None or source_col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return df[source_col].astype("object")


def coalesce_numeric_series(series_list, index=None) -> pd.Series:
    if len(series_list) == 0:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(np.nan, index=index, dtype=float)

    out = pd.Series(np.nan, index=series_list[0].index if index is None else index, dtype=float)

    for series_obj in series_list:
        if isinstance(series_obj, pd.Series):
            candidate = pd.to_numeric(series_obj, errors="coerce")
        else:
            candidate = pd.Series(series_obj, index=out.index, dtype=float)
        out = out.where(out.notna(), candidate)

    return out


def same_tick_flag(left_tick: pd.Series, right_tick: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left_tick, errors="coerce")
    right_num = pd.to_numeric(right_tick, errors="coerce")
    return (left_num.notna() & right_num.notna() & (left_num == right_num)).astype(int)


def mismatch_flag_same_tick(
    left_series: pd.Series,
    right_series: pd.Series,
    left_tick: pd.Series,
    right_tick: pd.Series,
) -> pd.Series:
    left_num = pd.to_numeric(left_series, errors="coerce")
    right_num = pd.to_numeric(right_series, errors="coerce")
    same_tick = same_tick_flag(left_tick, right_tick)

    return (
        (same_tick == 1)
        & left_num.notna()
        & right_num.notna()
        & (left_num != right_num)
    ).astype(int)


def resolve_signal_columns(df: pd.DataFrame) -> dict:
    resolved = {}

    for signal_name, source_list in CANONICAL_SIGNAL_SOURCES.items():
        resolved[signal_name] = first_existing_exact_column(df, source_list)

    return resolved


def add_standard_signal_columns(df: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    out = df.copy()

    out["schedule_raw_tick"] = numeric_from_resolved(out, resolved, "schedule_raw_tick")
    out["process_raw_tick"] = numeric_from_resolved(out, resolved, "process_raw_tick")
    out["simulation_raw_tick"] = numeric_from_resolved(out, resolved, "simulation_raw_tick")
    out["heater_raw_tick"] = numeric_from_resolved(out, resolved, "heater_raw_tick")
    out["substrate_raw_tick"] = numeric_from_resolved(out, resolved, "substrate_raw_tick")
    out["effusion_raw_tick"] = numeric_from_resolved(out, resolved, "effusion_raw_tick")
    out["battery_raw_tick"] = numeric_from_resolved(out, resolved, "battery_raw_tick")
    out["power_bus_raw_tick"] = numeric_from_resolved(out, resolved, "power_bus_raw_tick")
    out["solar_array_raw_tick"] = numeric_from_resolved(out, resolved, "solar_array_raw_tick")
    out["array_gimbal_raw_tick"] = numeric_from_resolved(out, resolved, "array_gimbal_raw_tick")
    out["battery_thermal_raw_tick"] = numeric_from_resolved(out, resolved, "battery_thermal_raw_tick")
    out["cryo_panel_raw_tick"] = numeric_from_resolved(out, resolved, "cryo_panel_raw_tick")
    out["radiator_raw_tick"] = numeric_from_resolved(out, resolved, "radiator_raw_tick")
    out["source_inventory_raw_tick"] = numeric_from_resolved(out, resolved, "source_inventory_raw_tick")

    out["schedule_controlling_job_index"] = numeric_from_resolved(out, resolved, "schedule_controlling_job_index")
    out["schedule_phase_code"] = numeric_from_resolved(out, resolved, "schedule_phase_code")
    out["schedule_mbe_flag"] = numeric_from_resolved(out, resolved, "schedule_mbe_flag")
    out["schedule_mbe_on"] = numeric_from_resolved(out, resolved, "schedule_mbe_on")
    out["schedule_substrate_on"] = numeric_from_resolved(out, resolved, "schedule_substrate_on")
    out["schedule_phase_ready_for_execution"] = numeric_from_resolved(out, resolved, "schedule_phase_ready_for_execution")
    out["schedule_deposition_requested"] = numeric_from_resolved(out, resolved, "schedule_deposition_requested")
    out["schedule_aborted_active"] = numeric_from_resolved(out, resolved, "schedule_aborted_active")
    out["schedule_live_active"] = numeric_from_resolved(out, resolved, "schedule_live_active")
    out["schedule_warmup_active"] = numeric_from_resolved(out, resolved, "schedule_warmup_active")
    out["schedule_cooldown_active"] = numeric_from_resolved(out, resolved, "schedule_cooldown_active")
    out["schedule_thermal_prep_active"] = numeric_from_resolved(out, resolved, "schedule_thermal_prep_active")

    out["simulation_job_failed"] = numeric_from_resolved(out, resolved, "simulation_job_failed")
    out["simulation_total_power_requested_w"] = numeric_from_resolved(out, resolved, "simulation_total_power_requested_w")
    out["simulation_total_power_granted_w"] = numeric_from_resolved(out, resolved, "simulation_total_power_granted_w")
    out["simulation_total_power_generated_w"] = numeric_from_resolved(out, resolved, "simulation_total_power_generated_w")
    out["simulation_solar_output_w"] = numeric_from_resolved(out, resolved, "simulation_solar_output_w")
    out["simulation_bus_remaining_w"] = numeric_from_resolved(out, resolved, "simulation_bus_remaining_w")
    out["simulation_status"] = object_from_resolved(out, resolved, "simulation_status")
    out["battery_capacity_wh_logged"] = numeric_from_resolved(out, resolved, "battery_capacity_wh_logged")

    out["substrate_job_index"] = numeric_from_resolved(out, resolved, "substrate_job_index")
    out["substrate_job_active"] = numeric_from_resolved(out, resolved, "substrate_job_active")
    out["substrate_control_on_logged"] = numeric_from_resolved(out, resolved, "substrate_control_on_logged")
    out["substrate_actual_temp_k"] = numeric_from_resolved(out, resolved, "substrate_actual_temp_k")
    out["substrate_target_temp_k_logged"] = numeric_from_resolved(out, resolved, "substrate_target_temp_k_logged")
    out["substrate_env_temp_k"] = numeric_from_resolved(out, resolved, "substrate_env_temp_k")
    out["substrate_solar_scale"] = numeric_from_resolved(out, resolved, "substrate_solar_scale")
    out["substrate_solar_absorbed_w"] = numeric_from_resolved(out, resolved, "substrate_solar_absorbed_w")
    out["substrate_requested_power_w_logged"] = numeric_from_resolved(out, resolved, "substrate_requested_power_w_logged")
    out["substrate_delivered_power_w"] = numeric_from_resolved(out, resolved, "substrate_delivered_power_w")
    out["substrate_power_loss_w"] = numeric_from_resolved(out, resolved, "substrate_power_loss_w")
    out["substrate_temp_miss_streak_logged"] = numeric_from_resolved(out, resolved, "substrate_temp_miss_streak_logged")
    out["substrate_c_j"] = numeric_from_resolved(out, resolved, "substrate_c_j")
    out["substrate_eps"] = numeric_from_resolved(out, resolved, "substrate_eps")
    out["substrate_h_wk"] = numeric_from_resolved(out, resolved, "substrate_h_wk")

    out["effusion_actual_temp_k"] = numeric_from_resolved(out, resolved, "effusion_actual_temp_k")
    out["effusion_target_temp_k_logged"] = numeric_from_resolved(out, resolved, "effusion_target_temp_k_logged")
    out["effusion_env_temp_k"] = numeric_from_resolved(out, resolved, "effusion_env_temp_k")
    out["effusion_solar_scale"] = numeric_from_resolved(out, resolved, "effusion_solar_scale")
    out["effusion_solar_absorbed_w"] = numeric_from_resolved(out, resolved, "effusion_solar_absorbed_w")
    out["effusion_heat_input_w"] = numeric_from_resolved(out, resolved, "effusion_heat_input_w")
    out["effusion_underflux_streak_logged"] = numeric_from_resolved(out, resolved, "effusion_underflux_streak_logged")
    out["effusion_temp_miss_streak_logged"] = numeric_from_resolved(out, resolved, "effusion_temp_miss_streak_logged")
    out["effusion_power_loss_w"] = numeric_from_resolved(out, resolved, "effusion_power_loss_w")
    out["effusion_power_net_w"] = numeric_from_resolved(out, resolved, "effusion_power_net_w")
    out["effusion_requested_power_w_logged"] = numeric_from_resolved(out, resolved, "effusion_requested_power_w_logged")
    out["effusion_delivered_power_w"] = numeric_from_resolved(out, resolved, "effusion_delivered_power_w")
    out["effusion_c_j"] = numeric_from_resolved(out, resolved, "effusion_c_j")
    out["effusion_h_wk"] = numeric_from_resolved(out, resolved, "effusion_h_wk")

    out["process_controlling_job_index"] = numeric_from_resolved(out, resolved, "process_controlling_job_index")
    out["process_mbe_on"] = numeric_from_resolved(out, resolved, "process_mbe_on")
    out["process_substrate_on"] = numeric_from_resolved(out, resolved, "process_substrate_on")
    out["process_raw_job_flux_cm2s"] = numeric_from_resolved(out, resolved, "process_raw_job_flux_cm2s")
    out["process_deposition_requested"] = numeric_from_resolved(out, resolved, "process_deposition_requested")
    out["process_phase_ready_for_execution"] = numeric_from_resolved(out, resolved, "process_phase_ready_for_execution")
    out["process_effusion_demand_w"] = numeric_from_resolved(out, resolved, "process_effusion_demand_w")
    out["process_substrate_demand_w"] = numeric_from_resolved(out, resolved, "process_substrate_demand_w")
    out["process_effusion_ready"] = numeric_from_resolved(out, resolved, "process_effusion_ready")
    out["process_wafer_ready"] = numeric_from_resolved(out, resolved, "process_wafer_ready")
    out["process_mbe_flag"] = numeric_from_resolved(out, resolved, "process_mbe_flag")
    out["process_effusion_delivered_w"] = numeric_from_resolved(out, resolved, "process_effusion_delivered_w")
    out["process_substrate_delivered_w"] = numeric_from_resolved(out, resolved, "process_substrate_delivered_w")

    out["solar_array_output_w"] = numeric_from_resolved(out, resolved, "solar_array_output_w")
    out["battery_charge_wh_raw"] = numeric_from_resolved(out, resolved, "battery_charge_wh_raw")
    out["battery_max_charge_w_logged"] = numeric_from_resolved(out, resolved, "battery_max_charge_w_logged")
    out["battery_max_discharge_w_logged"] = numeric_from_resolved(out, resolved, "battery_max_discharge_w_logged")
    out["battery_power_w_raw"] = numeric_from_resolved(out, resolved, "battery_power_w_raw")
    out["power_bus_solar_added_w"] = numeric_from_resolved(out, resolved, "power_bus_solar_added_w")
    out["power_bus_requested_w"] = numeric_from_resolved(out, resolved, "power_bus_requested_w")
    out["power_bus_granted_w"] = numeric_from_resolved(out, resolved, "power_bus_granted_w")

    out["effusion_requested_power_w"] = coalesce_numeric_series(
        [
            out["effusion_requested_power_w_logged"],
            out["process_effusion_demand_w"],
            numeric_or_nan(out, "expected_effusion_heater_cap_w"),
        ],
        index=out.index,
    )

    out["effusion_delivered_power_w_canonical"] = coalesce_numeric_series(
        [
            out["effusion_delivered_power_w"],
            out["process_effusion_delivered_w"],
        ],
        index=out.index,
    )

    out["substrate_target_temp_k"] = coalesce_numeric_series(
        [
            out["substrate_target_temp_k_logged"],
            numeric_or_nan(out, "expected_substrate_target_k"),
        ],
        index=out.index,
    )

    out["substrate_requested_power_w"] = coalesce_numeric_series(
        [
            out["substrate_requested_power_w_logged"],
            out["process_substrate_demand_w"],
            numeric_or_nan(out, "substrate_max_power_draw_w"),
        ],
        index=out.index,
    )

    out["substrate_delivered_power_w_canonical"] = coalesce_numeric_series(
        [
            out["substrate_delivered_power_w"],
            out["process_substrate_delivered_w"],
        ],
        index=out.index,
    )

    out["total_requested_power_w"] = coalesce_numeric_series(
        [
            out["simulation_total_power_requested_w"],
            out["power_bus_requested_w"],
            numeric_or_nan(out, "heater_bank_max_draw_w"),
        ],
        index=out.index,
    )

    out["total_delivered_power_w"] = coalesce_numeric_series(
        [
            out["simulation_total_power_granted_w"],
            out["power_bus_granted_w"],
        ],
        index=out.index,
    )

    out["solar_power_w_raw"] = coalesce_numeric_series(
        [
            out["solar_array_output_w"],
            out["simulation_solar_output_w"],
            out["power_bus_solar_added_w"],
        ],
        index=out.index,
    )

    out["actual_subjob_index"] = coalesce_numeric_series(
        [
            out["process_controlling_job_index"],
            out["schedule_controlling_job_index"],
            numeric_or_nan(out, "subjob_index"),
        ],
        index=out.index,
    )

    out["actual_phase_code"] = coalesce_numeric_series(
        [
            out["schedule_phase_code"],
            numeric_or_nan(out, "phase_code_expected"),
        ],
        index=out.index,
    )

    out["actual_mbe_flag"] = coalesce_numeric_series(
        [
            out["process_mbe_flag"],
            out["schedule_mbe_flag"],
            numeric_or_nan(out, "expected_mbe_on"),
        ],
        index=out.index,
    )

    out["actual_mbe_on"] = coalesce_numeric_series(
        [
            out["process_mbe_on"],
            out["schedule_mbe_on"],
            numeric_or_nan(out, "expected_mbe_on"),
        ],
        index=out.index,
    )

    out["actual_substrate_on"] = coalesce_numeric_series(
        [
            out["process_substrate_on"],
            out["schedule_substrate_on"],
            numeric_or_nan(out, "expected_substrate_on"),
        ],
        index=out.index,
    )

    out["actual_phase_ready_for_execution"] = coalesce_numeric_series(
        [
            out["process_phase_ready_for_execution"],
            out["schedule_phase_ready_for_execution"],
        ],
        index=out.index,
    )

    out["actual_deposition_requested"] = coalesce_numeric_series(
        [
            out["process_deposition_requested"],
            out["schedule_deposition_requested"],
        ],
        index=out.index,
    )

    out["phase_name_resolved"] = out["actual_phase_code"].map(PHASE_CODE_TO_NAME)
    if "phase_name" in out.columns:
        out["phase_name_resolved"] = out["phase_name_resolved"].where(
            out["phase_name_resolved"].notna(),
            out["phase_name"].astype("object"),
        )

    return out


def ratio_in_fraction_band(actual: pd.Series, target: pd.Series, tolerance_fraction: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    target_num = pd.to_numeric(target, errors="coerce")
    tolerance_num = pd.to_numeric(tolerance_fraction, errors="coerce")

    ratio = pd.Series(safe_divide(actual_num, target_num), index=actual_num.index)
    lower = tolerance_num.clip(lower=0)
    upper = 1.0 / lower.replace(0, np.nan)

    return (
        ratio.notna()
        & lower.notna()
        & upper.notna()
        & (ratio >= lower)
        & (ratio <= upper)
    ).astype(int)


def compute_segment_age(series: pd.Series) -> pd.Series:
    marker = series.astype("object").fillna("missing")
    group_id = marker.ne(marker.shift()).cumsum()
    return group_id.groupby(group_id).cumcount()


def add_rule_features_one_run(df: pd.DataFrame):
    resolved = resolve_signal_columns(df)
    out = add_standard_signal_columns(df, resolved)

    battery_capacity_wh_cfg = numeric_or_nan(out, "battery_capacity_wh")
    solar_base_input_w_cfg = numeric_or_nan(out, "solar_base_input_w")
    ready_band_k_cfg = numeric_or_nan(out, "ready_band_k")
    effusion_min_flux_fraction_cfg = numeric_or_nan(out, "effusion_min_flux_fraction")
    effusion_temp_tolerance_fraction_cfg = numeric_or_nan(out, "effusion_temp_tolerance_fraction")

    out["battery_charge_fraction"] = pd.Series(
        safe_divide(out["battery_charge_wh_raw"], battery_capacity_wh_cfg),
        index=out.index,
    )

    out["solar_fraction_of_base"] = pd.Series(
        safe_divide(out["solar_power_w_raw"], solar_base_input_w_cfg),
        index=out.index,
    )

    out["effusion_flux_ratio"] = pd.Series(
        safe_divide(out["effusion_delivered_power_w_canonical"], out["effusion_requested_power_w"]),
        index=out.index,
    )

    out["effusion_temp_ratio"] = pd.Series(
        safe_divide(out["effusion_actual_temp_k"], out["effusion_target_temp_k_logged"]),
        index=out.index,
    )

    out["substrate_temp_error_k"] = out["substrate_actual_temp_k"] - out["substrate_target_temp_k"]

    out["wafer_in_band"] = (
        out["substrate_temp_error_k"].notna()
        & ready_band_k_cfg.notna()
        & out["substrate_temp_error_k"].abs().le(ready_band_k_cfg)
    ).astype(int)

    out["wafer_below_band"] = (
        out["substrate_temp_error_k"].notna()
        & ready_band_k_cfg.notna()
        & (out["substrate_temp_error_k"] < -ready_band_k_cfg)
    ).astype(int)

    out["wafer_above_band"] = (
        out["substrate_temp_error_k"].notna()
        & ready_band_k_cfg.notna()
        & (out["substrate_temp_error_k"] > ready_band_k_cfg)
    ).astype(int)

    out["effusion_temp_in_band"] = ratio_in_fraction_band(
        out["effusion_actual_temp_k"],
        out["effusion_target_temp_k_logged"],
        effusion_temp_tolerance_fraction_cfg,
    )

    out["power_deficit_total_w"] = (
        pd.to_numeric(out["total_requested_power_w"], errors="coerce")
        - pd.to_numeric(out["total_delivered_power_w"], errors="coerce")
    ).clip(lower=0)

    out["power_deficit_effusion_w"] = (
        pd.to_numeric(out["effusion_requested_power_w"], errors="coerce")
        - pd.to_numeric(out["effusion_delivered_power_w_canonical"], errors="coerce")
    ).clip(lower=0)

    out["power_deficit_substrate_w"] = (
        pd.to_numeric(out["substrate_requested_power_w"], errors="coerce")
        - pd.to_numeric(out["substrate_delivered_power_w_canonical"], errors="coerce")
    ).clip(lower=0)

    out["is_recipe_active"] = numeric_or_nan(out, "subjob_index").notna().astype(int)

    out["is_live_phase"] = (
        (numeric_or_nan(out, "is_growth_like").fillna(0) == 1)
        | (pd.to_numeric(out["actual_mbe_on"], errors="coerce").fillna(0) == 1)
    ).astype(int)

    out["is_substrate_control_expected"] = (
        pd.to_numeric(out["actual_substrate_on"], errors="coerce").fillna(0) == 1
    ).astype(int)

    out["time_since_subjob_start"] = numeric_or_nan(out, "ticks_into_subjob")
    out["time_until_subjob_end"] = numeric_or_nan(out, "ticks_remaining_in_subjob")

    out["phase_name_resolved"] = object_or_na(out, "phase_name_resolved")
    out["time_since_phase_transition"] = compute_segment_age(out["phase_name_resolved"])

    out["live_underflux_flag"] = (
        (out["is_live_phase"] == 1)
        & out["effusion_flux_ratio"].notna()
        & (out["effusion_flux_ratio"] < effusion_min_flux_fraction_cfg)
    ).astype(int)

    out["live_effusion_temp_miss_flag"] = (
        (out["is_live_phase"] == 1)
        & out["effusion_actual_temp_k"].notna()
        & out["effusion_target_temp_k_logged"].notna()
        & (out["effusion_temp_in_band"] == 0)
    ).astype(int)

    out["wafer_temp_miss_flag"] = (
        (out["is_substrate_control_expected"] == 1)
        & out["substrate_actual_temp_k"].notna()
        & out["substrate_target_temp_k"].notna()
        & (out["wafer_in_band"] == 0)
    ).astype(int)

    out["power_starvation_flag"] = (
        (out["is_recipe_active"] == 1)
        & (
            (out["power_deficit_total_w"] > 0)
            | (out["power_deficit_effusion_w"] > 0)
            | (out["power_deficit_substrate_w"] > 0)
        )
    ).astype(int)

    out["live_underflux_streak"] = consecutive_count_from_flag(out["live_underflux_flag"])
    out["live_effusion_temp_miss_streak"] = consecutive_count_from_flag(out["live_effusion_temp_miss_flag"])
    out["wafer_temp_miss_streak"] = consecutive_count_from_flag(out["wafer_temp_miss_flag"])
    out["power_starvation_streak"] = consecutive_count_from_flag(out["power_starvation_flag"])

    flux_ready = (
        (out["is_live_phase"] == 0)
        | (
            out["effusion_flux_ratio"].notna()
            & (out["effusion_flux_ratio"] >= effusion_min_flux_fraction_cfg)
        )
    )

    effusion_temp_ready = (
        (out["is_live_phase"] == 0)
        | (
            out["effusion_actual_temp_k"].notna()
            & out["effusion_target_temp_k_logged"].notna()
            & (out["effusion_temp_in_band"] == 1)
        )
    )

    wafer_ready = (
        (out["is_substrate_control_expected"] == 0)
        | (
            out["substrate_actual_temp_k"].notna()
            & out["substrate_target_temp_k"].notna()
            & (out["wafer_in_band"] == 1)
        )
    )

    expected_effusion_heater_cap_w = numeric_or_nan(out, "expected_effusion_heater_cap_w")

    state = np.where(
        out["is_recipe_active"] == 0,
        "no_recipe",
        np.where(
            out["is_live_phase"] == 1,
            np.where(flux_ready & effusion_temp_ready & wafer_ready, "live_ready", "live_unready"),
            np.where(
                (out["is_substrate_control_expected"] == 1)
                | (expected_effusion_heater_cap_w.fillna(0) > 0),
                np.where(effusion_temp_ready & wafer_ready, "ready", "warming"),
                "inactive",
            ),
        ),
    )

    out["candidate_readiness_state"] = pd.Series(state, index=out.index, dtype="object")

    state_code_map = {
        "no_recipe": 0,
        "inactive": 1,
        "warming": 2,
        "ready": 3,
        "live_unready": 4,
        "live_ready": 5,
    }

    out["candidate_readiness_code"] = out["candidate_readiness_state"].map(state_code_map).astype("Int64")
    out["consecutive_warming_ticks"] = consecutive_count_from_flag((out["candidate_readiness_state"] == "warming").astype(int))
    out["consecutive_live_unready_ticks"] = consecutive_count_from_flag((out["candidate_readiness_state"] == "live_unready").astype(int))

    out["process_schedule_same_raw_tick_flag"] = same_tick_flag(
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_heater_same_raw_tick_flag"] = same_tick_flag(
        out["process_raw_tick"],
        out["heater_raw_tick"],
    )

    out["process_schedule_job_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_controlling_job_index"],
        out["schedule_controlling_job_index"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_schedule_mbe_flag_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_mbe_flag"],
        out["schedule_mbe_flag"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_schedule_mbe_on_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_mbe_on"],
        out["schedule_mbe_on"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_schedule_substrate_on_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_substrate_on"],
        out["schedule_substrate_on"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_schedule_phase_ready_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_phase_ready_for_execution"],
        out["schedule_phase_ready_for_execution"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_schedule_deposition_requested_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_deposition_requested"],
        out["schedule_deposition_requested"],
        out["process_raw_tick"],
        out["schedule_raw_tick"],
    )

    out["process_heater_eff_requested_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_effusion_demand_w"],
        out["effusion_requested_power_w_logged"],
        out["process_raw_tick"],
        out["heater_raw_tick"],
    )

    out["process_heater_eff_delivered_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_effusion_delivered_w"],
        out["effusion_delivered_power_w"],
        out["process_raw_tick"],
        out["heater_raw_tick"],
    )

    out["process_heater_sub_requested_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_substrate_demand_w"],
        out["substrate_requested_power_w_logged"],
        out["process_raw_tick"],
        out["heater_raw_tick"],
    )

    out["process_heater_sub_delivered_mismatch_flag"] = mismatch_flag_same_tick(
        out["process_substrate_delivered_w"],
        out["substrate_delivered_power_w"],
        out["process_raw_tick"],
        out["heater_raw_tick"],
    )

    out["log_disagreement_flag"] = (
        (out["process_schedule_job_mismatch_flag"] == 1)
        | (out["process_schedule_mbe_flag_mismatch_flag"] == 1)
        | (out["process_schedule_mbe_on_mismatch_flag"] == 1)
        | (out["process_schedule_substrate_on_mismatch_flag"] == 1)
        | (out["process_schedule_phase_ready_mismatch_flag"] == 1)
        | (out["process_schedule_deposition_requested_mismatch_flag"] == 1)
        | (out["process_heater_eff_requested_mismatch_flag"] == 1)
        | (out["process_heater_eff_delivered_mismatch_flag"] == 1)
        | (out["process_heater_sub_requested_mismatch_flag"] == 1)
        | (out["process_heater_sub_delivered_mismatch_flag"] == 1)
    ).astype(int)

    return out, resolved


def add_rule_features_all_runs(df: pd.DataFrame):
    feature_pieces = []
    sample_resolved_map = None

    for run_id, run_df in df.groupby("run_id", sort=False, dropna=False):
        run_df = run_df.sort_values("canonical_tick").reset_index(drop=True).copy()
        engineered_df, resolved = add_rule_features_one_run(run_df)
        feature_pieces.append(engineered_df)

        if sample_resolved_map is None:
            sample_resolved_map = resolved

    if len(feature_pieces) == 0:
        return pd.DataFrame(), {}

    out = pd.concat(feature_pieces, ignore_index=True, sort=False)
    out = out.sort_values(["config_label", "job_name", "canonical_tick"], na_position="last").reset_index(drop=True)
    return out, sample_resolved_map


# build the sample from the enriched all runs dataframe
# this avoids the earlier failure where sample_tick_df did not contain tracker config columns

if all_runs_tick_df.empty:
    raise ValueError("all_runs_tick_df is empty so features cannot be built")

sample_run_id_for_features = all_runs_tick_df["run_id"].dropna().astype(str).iloc[0]
sample_tick_with_context_df = (
    all_runs_tick_df.loc[all_runs_tick_df["run_id"].astype(str) == sample_run_id_for_features]
    .sort_values("canonical_tick")
    .reset_index(drop=True)
    .copy()
)

sample_features_df, sample_resolved_map = add_rule_features_one_run(sample_tick_with_context_df)
all_runs_features_df, _ = add_rule_features_all_runs(all_runs_tick_df)

all_runs_features_parquet_path = OUTPUT_ROOT / "all_runs_features_df.parquet"
sample_features_parquet_path = OUTPUT_ROOT / "sample_features_df.parquet"

if not all_runs_features_df.empty:
    all_runs_features_df.to_parquet(all_runs_features_parquet_path, index=False)

if not sample_features_df.empty:
    sample_features_df.to_parquet(sample_features_parquet_path, index=False)

print("sample run id for features", sample_run_id_for_features)
print("sample feature df shape", sample_features_df.shape)
print("all runs feature df shape", all_runs_features_df.shape)
print("all runs features parquet path", all_runs_features_parquet_path)
print("sample features parquet path", sample_features_parquet_path)

print("sample resolved signals")
display(
    pd.DataFrame(
        {
            "signal_name": list(sample_resolved_map.keys()),
            "source_column": list(sample_resolved_map.values()),
        }
    )
)

show_cols = [
    "run_id",
    "canonical_tick",
    "process_raw_tick",
    "schedule_raw_tick",
    "heater_raw_tick",
    "simulation_raw_tick",
    "subjob_index",
    "phase_name_resolved",
    "effusion_actual_temp_k",
    "effusion_target_temp_k_logged",
    "effusion_requested_power_w",
    "effusion_delivered_power_w_canonical",
    "effusion_flux_ratio",
    "effusion_temp_ratio",
    "substrate_actual_temp_k",
    "substrate_target_temp_k",
    "substrate_requested_power_w",
    "substrate_delivered_power_w_canonical",
    "substrate_temp_error_k",
    "wafer_in_band",
    "live_underflux_streak",
    "live_effusion_temp_miss_streak",
    "wafer_temp_miss_streak",
    "candidate_readiness_state",
    "process_schedule_same_raw_tick_flag",
    "process_heater_same_raw_tick_flag",
    "process_schedule_job_mismatch_flag",
    "process_heater_eff_requested_mismatch_flag",
    "process_heater_eff_delivered_mismatch_flag",
    "process_heater_sub_requested_mismatch_flag",
    "process_heater_sub_delivered_mismatch_flag",
    "log_disagreement_flag",
]
show_cols = [col for col in show_cols if col in sample_features_df.columns]

display(sample_features_df[show_cols].head(30))

# funnel step 1
# mark runs that fail because the raw effusion underflux or undertemp streak hits a normal cap above one
#
# important logic
# this step only checks normal cap hits
# cap one cases are intentionally skipped here
# those should be handled by a separate special case cell
#
# funnel rule
# only runs that are still alive are inspected in this step
# once a run is marked as failed here it should not be inspected by later funnel steps
#
# data rule
# this step uses the merged per tick dataframe built earlier
# the true failure tick is the canonical tick where the raw effusion streak first equals the config cap

def valid_integer_cap(value):
    """
    convert one tracker cap value into a valid positive integer cap

    input
    tracker cap value

    output
    integer cap or nan
    """
    value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(value_num):
        return np.nan

    rounded_value = int(round(float(value_num)))

    if rounded_value < 1:
        return np.nan

    if not np.isclose(float(value_num), rounded_value):
        return np.nan

    return rounded_value


def initialize_or_update_run_funnel_df(run_manifest: pd.DataFrame, existing_funnel_df=None) -> pd.DataFrame:
    """
    build the run level funnel status table or add any missing columns

    input
    run_manifest one row per run
    existing_funnel_df optional existing funnel table

    output
    run level funnel dataframe
    """
    base_df = run_manifest[
        [
            "run_id",
            "config_id",
            "config_label",
            "config_numeric_id",
            "config_row_id",
            "config_folder_name",
            "job_name",
            "job_file_path",
            "run_dir",
        ]
    ].drop_duplicates(subset=["run_id"]).copy()

    if existing_funnel_df is None:
        funnel_df = base_df.copy()
    else:
        funnel_df = base_df.merge(existing_funnel_df, on="run_id", how="left", suffixes=("", "_old"))

        # keep the base metadata columns from the fresh manifest version
        old_suffix_cols = [c for c in funnel_df.columns if c.endswith("_old")]
        if old_suffix_cols:
            funnel_df = funnel_df.drop(columns=old_suffix_cols)

    default_columns = {
        "funnel_failed": 0,
        "funnel_failure_reason": pd.NA,
        "funnel_failure_tick": np.nan,
        "funnel_failure_step": pd.NA,
        "failed_effusion_normal_cap_hit": 0,
        "failed_effusion_underflux_cap_hit": 0,
        "failed_effusion_undertemp_cap_hit": 0,
    }

    for col, default_value in default_columns.items():
        if col not in funnel_df.columns:
            funnel_df[col] = default_value

    funnel_df["funnel_failed"] = pd.to_numeric(funnel_df["funnel_failed"], errors="coerce").fillna(0).astype(int)
    funnel_df["failed_effusion_normal_cap_hit"] = pd.to_numeric(funnel_df["failed_effusion_normal_cap_hit"], errors="coerce").fillna(0).astype(int)
    funnel_df["failed_effusion_underflux_cap_hit"] = pd.to_numeric(funnel_df["failed_effusion_underflux_cap_hit"], errors="coerce").fillna(0).astype(int)
    funnel_df["failed_effusion_undertemp_cap_hit"] = pd.to_numeric(funnel_df["failed_effusion_undertemp_cap_hit"], errors="coerce").fillna(0).astype(int)

    return funnel_df.sort_values(["config_label", "job_name"]).reset_index(drop=True)


# initialize the run level funnel table if it does not already exist
if "run_funnel_df" in globals():
    run_funnel_df = initialize_or_update_run_funnel_df(run_manifest, run_funnel_df)
else:
    run_funnel_df = initialize_or_update_run_funnel_df(run_manifest)

eligible_run_ids = set(
    run_funnel_df.loc[run_funnel_df["funnel_failed"] == 0, "run_id"].astype(str).tolist()
)

required_cols = [
    "run_id",
    "config_label",
    "job_name",
    "canonical_tick",
    "effusion_underflux_streak_cap",
    "effusion_undertemp_streak_cap",
]

missing_required_cols = [col for col in required_cols if col not in all_runs_tick_df.columns]
if missing_required_cols:
    raise ValueError(f"missing required columns in all_runs_tick_df {missing_required_cols}")

# these raw effusion streak columns should have been carried in from the merged effusion log
effusion_streak_cols = {
    "underflux": "effusion__underflux_streak",
    "undertemp": "effusion__temp_miss_streak",
}

for cause_name, cause_col in effusion_streak_cols.items():
    if cause_col not in all_runs_tick_df.columns:
        raise ValueError(f"missing expected effusion streak column {cause_col}")

candidate_df = all_runs_tick_df.loc[all_runs_tick_df["run_id"].astype(str).isin(eligible_run_ids)].copy()

candidate_df["canonical_tick"] = pd.to_numeric(candidate_df["canonical_tick"], errors="coerce")
candidate_df["effusion_underflux_streak_cap"] = pd.to_numeric(candidate_df["effusion_underflux_streak_cap"], errors="coerce")
candidate_df["effusion_undertemp_streak_cap"] = pd.to_numeric(candidate_df["effusion_undertemp_streak_cap"], errors="coerce")
candidate_df["effusion__underflux_streak"] = pd.to_numeric(candidate_df["effusion__underflux_streak"], errors="coerce")
candidate_df["effusion__temp_miss_streak"] = pd.to_numeric(candidate_df["effusion__temp_miss_streak"], errors="coerce")

effusion_cap_hit_rows = []
bad_cap_rows = []
cap_one_rows_skipped = []

for run_id, run_df in candidate_df.groupby("run_id", dropna=False):
    run_df = run_df.sort_values("canonical_tick").copy()

    run_meta = run_df.iloc[0]

    underflux_cap_raw = run_meta.get("effusion_underflux_streak_cap", np.nan)
    undertemp_cap_raw = run_meta.get("effusion_undertemp_streak_cap", np.nan)

    underflux_cap = valid_integer_cap(underflux_cap_raw)
    undertemp_cap = valid_integer_cap(undertemp_cap_raw)

    # underflux cause
    if pd.isna(underflux_cap):
        bad_cap_rows.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "underflux",
                "cap_raw_value": underflux_cap_raw,
            }
        )
    elif underflux_cap == 1:
        cap_one_rows_skipped.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "underflux",
                "cap_value": underflux_cap,
            }
        )
    else:
        underflux_hit_df = run_df.loc[run_df["effusion__underflux_streak"].eq(underflux_cap)].copy()

        if not underflux_hit_df.empty:
            first_row = underflux_hit_df.iloc[0]

            effusion_cap_hit_rows.append(
                {
                    "failure_reason": "effusion_underflux_cap_hit",
                    "run_id": run_id,
                    "config_label": run_meta["config_label"],
                    "job_name": run_meta["job_name"],
                    "failure_tick": first_row["canonical_tick"],
                    "effusion_underflux_cap": underflux_cap,
                    "effusion_undertemp_cap": undertemp_cap,
                    "effusion_underflux_streak_value": first_row["effusion__underflux_streak"],
                    "effusion_undertemp_streak_value": first_row.get("effusion__temp_miss_streak", np.nan),
                    "effusion_raw_tick": first_row.get("effusion__tick", np.nan),
                }
            )

    # undertemp cause
    if pd.isna(undertemp_cap):
        bad_cap_rows.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "undertemp",
                "cap_raw_value": undertemp_cap_raw,
            }
        )
    elif undertemp_cap == 1:
        cap_one_rows_skipped.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "undertemp",
                "cap_value": undertemp_cap,
            }
        )
    else:
        undertemp_hit_df = run_df.loc[run_df["effusion__temp_miss_streak"].eq(undertemp_cap)].copy()

        if not undertemp_hit_df.empty:
            first_row = undertemp_hit_df.iloc[0]

            effusion_cap_hit_rows.append(
                {
                    "failure_reason": "effusion_undertemp_cap_hit",
                    "run_id": run_id,
                    "config_label": run_meta["config_label"],
                    "job_name": run_meta["job_name"],
                    "failure_tick": first_row["canonical_tick"],
                    "effusion_underflux_cap": underflux_cap,
                    "effusion_undertemp_cap": undertemp_cap,
                    "effusion_underflux_streak_value": first_row.get("effusion__underflux_streak", np.nan),
                    "effusion_undertemp_streak_value": first_row["effusion__temp_miss_streak"],
                    "effusion_raw_tick": first_row.get("effusion__tick", np.nan),
                }
            )

if len(effusion_cap_hit_rows) == 0:
    effusion_cap_hits_df = pd.DataFrame(
        columns=[
            "failure_reason",
            "run_id",
            "config_label",
            "job_name",
            "failure_tick",
            "effusion_underflux_cap",
            "effusion_undertemp_cap",
            "effusion_underflux_streak_value",
            "effusion_undertemp_streak_value",
            "effusion_raw_tick",
        ]
    )
else:
    effusion_cap_hits_df = pd.DataFrame(effusion_cap_hit_rows).sort_values(
        ["config_label", "job_name", "failure_tick", "failure_reason"],
        na_position="last",
    ).reset_index(drop=True)

# choose the first effusion failure per run for the funnel
# this keeps one first failure stamp per run even if both effusion causes appear
if effusion_cap_hits_df.empty:
    effusion_first_failures_df = effusion_cap_hits_df.copy()
else:
    effusion_first_failures_df = (
        effusion_cap_hits_df
        .sort_values(["run_id", "failure_tick", "failure_reason"], na_position="last")
        .drop_duplicates(subset=["run_id"], keep="first")
        .reset_index(drop=True)
    )

# update the run level funnel table
if not effusion_first_failures_df.empty:
    for _, fail_row in effusion_first_failures_df.iterrows():
        run_id = fail_row["run_id"]
        failure_reason = fail_row["failure_reason"]
        failure_tick = fail_row["failure_tick"]

        mask = run_funnel_df["run_id"].astype(str).eq(str(run_id)) & run_funnel_df["funnel_failed"].eq(0)

        run_funnel_df.loc[mask, "funnel_failed"] = 1
        run_funnel_df.loc[mask, "funnel_failure_reason"] = failure_reason
        run_funnel_df.loc[mask, "funnel_failure_tick"] = failure_tick
        run_funnel_df.loc[mask, "funnel_failure_step"] = "effusion_normal_cap_hit"
        run_funnel_df.loc[mask, "failed_effusion_normal_cap_hit"] = 1

        if failure_reason == "effusion_underflux_cap_hit":
            run_funnel_df.loc[mask, "failed_effusion_underflux_cap_hit"] = 1

        if failure_reason == "effusion_undertemp_cap_hit":
            run_funnel_df.loc[mask, "failed_effusion_undertemp_cap_hit"] = 1

# push the current run level funnel status back onto the full per tick dataframe
run_failed_map = run_funnel_df.set_index("run_id")["funnel_failed"].to_dict()
run_failure_reason_map = run_funnel_df.set_index("run_id")["funnel_failure_reason"].to_dict()
run_failure_tick_map = run_funnel_df.set_index("run_id")["funnel_failure_tick"].to_dict()
run_failure_step_map = run_funnel_df.set_index("run_id")["funnel_failure_step"].to_dict()

all_runs_tick_df["funnel_failed"] = (
    all_runs_tick_df["run_id"].map(run_failed_map).fillna(0).astype(int)
)
all_runs_tick_df["funnel_failure_reason"] = all_runs_tick_df["run_id"].map(run_failure_reason_map)
all_runs_tick_df["funnel_failure_tick"] = all_runs_tick_df["run_id"].map(run_failure_tick_map)
all_runs_tick_df["funnel_failure_step"] = all_runs_tick_df["run_id"].map(run_failure_step_map)
all_runs_tick_df["funnel_is_failure_tick"] = (
    all_runs_tick_df["funnel_failed"].eq(1)
    & pd.to_numeric(all_runs_tick_df["canonical_tick"], errors="coerce").eq(
        pd.to_numeric(all_runs_tick_df["funnel_failure_tick"], errors="coerce")
    )
).astype(int)

print("funnel step name", "effusion_normal_cap_hit")
print("runs eligible at step start", len(eligible_run_ids))
print("runs newly failed in this step", len(effusion_first_failures_df))
print("runs still alive after this step", int((run_funnel_df["funnel_failed"] == 0).sum()))

if len(bad_cap_rows) > 0:
    print("these runs were skipped for one cause because the tracker cap was not a valid whole number")
    display(
        pd.DataFrame(bad_cap_rows)
        .drop_duplicates(subset=["config_label", "cap_type"])
        .sort_values(["config_label", "cap_type"])
        .reset_index(drop=True)
    )

if len(cap_one_rows_skipped) > 0:
    print("these cap one causes were intentionally skipped here and should be checked only in the special case cell")
    display(
        pd.DataFrame(cap_one_rows_skipped)
        .drop_duplicates(subset=["config_label", "cap_type"])
        .sort_values(["config_label", "cap_type"])
        .reset_index(drop=True)
    )

if effusion_cap_hits_df.empty:
    print("no raw effusion rows hit any normal streak cap above one among the currently alive runs")
else:
    print("exact merged tick rows where the raw effusion streak first hit a normal cap above one")
    print("rows shown", len(effusion_cap_hits_df))
    display(effusion_cap_hits_df)

    print("first effusion failure stamped into the funnel for each affected run")
    display(effusion_first_failures_df)

print("current run level funnel status")
display(
    run_funnel_df.sort_values(["config_label", "job_name"]).reset_index(drop=True)
)

print("sample of the full per tick dataframe after funnel annotation")
display(
    all_runs_tick_df[
        [
            "run_id",
            "config_label",
            "job_name",
            "canonical_tick",
            "funnel_failed",
            "funnel_failure_reason",
            "funnel_failure_tick",
            "funnel_failure_step",
            "funnel_is_failure_tick",
        ]
    ]
    .sort_values(["config_label", "job_name", "canonical_tick"])
    .head(30)
)

# funnel step 2
# mark runs that fail because an effusion cap of one is violated on the same live tick
#
# important logic
# this step only checks the special cap one case
# normal caps above one were already handled in the previous funnel step
#
# funnel rule
# only runs that are still alive are inspected in this step
# once a run is marked as failed here it should not be inspected by later funnel steps
#
# data rule
# this step uses the merged same tick dataframe
# processstate and simulationengine were kept on the native tick axis
# so the true failure tick here is the first live tick where the same tick ratio violates the config threshold

# add the run level columns for this funnel step if they do not exist yet
cap_one_default_columns = {
    "failed_effusion_cap_one_hit": 0,
    "failed_effusion_underflux_cap_one_hit": 0,
    "failed_effusion_undertemp_cap_one_hit": 0,
}

for col, default_value in cap_one_default_columns.items():
    if col not in run_funnel_df.columns:
        run_funnel_df[col] = default_value

run_funnel_df["failed_effusion_cap_one_hit"] = pd.to_numeric(
    run_funnel_df["failed_effusion_cap_one_hit"],
    errors="coerce",
).fillna(0).astype(int)

run_funnel_df["failed_effusion_underflux_cap_one_hit"] = pd.to_numeric(
    run_funnel_df["failed_effusion_underflux_cap_one_hit"],
    errors="coerce",
).fillna(0).astype(int)

run_funnel_df["failed_effusion_undertemp_cap_one_hit"] = pd.to_numeric(
    run_funnel_df["failed_effusion_undertemp_cap_one_hit"],
    errors="coerce",
).fillna(0).astype(int)

eligible_run_ids = set(
    run_funnel_df.loc[run_funnel_df["funnel_failed"] == 0, "run_id"].astype(str).tolist()
)

required_cols = [
    "run_id",
    "config_label",
    "job_name",
    "canonical_tick",
    "effusion_underflux_streak_cap",
    "effusion_undertemp_streak_cap",
    "effusion_min_flux_fraction",
    "effusion_temp_tolerance_fraction",
    "process_state__mbe_flag",
    "effusion__heatInput_w",
    "process_state__effusionDemand_W",
    "effusion__act_temp_K",
    "effusion__target_temp_K",
    "effusion__underflux_streak",
    "effusion__temp_miss_streak",
]

missing_required_cols = [col for col in required_cols if col not in all_runs_tick_df.columns]
if missing_required_cols:
    raise ValueError(f"missing required columns in all_runs_tick_df {missing_required_cols}")

candidate_df = all_runs_tick_df.loc[
    all_runs_tick_df["run_id"].astype(str).isin(eligible_run_ids)
].copy()

numeric_cols = [
    "canonical_tick",
    "effusion_underflux_streak_cap",
    "effusion_undertemp_streak_cap",
    "effusion_min_flux_fraction",
    "effusion_temp_tolerance_fraction",
    "process_state__mbe_flag",
    "effusion__heatInput_w",
    "process_state__effusionDemand_W",
    "effusion__act_temp_K",
    "effusion__target_temp_K",
    "effusion__underflux_streak",
    "effusion__temp_miss_streak",
]

for col in numeric_cols:
    candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")

candidate_df["same_tick_live_flag"] = (
    candidate_df["process_state__mbe_flag"].fillna(0).eq(1)
).astype(int)

candidate_df["effusion_flux_ratio_same_tick"] = np.nan
valid_flux_mask = (
    candidate_df["effusion__heatInput_w"].notna()
    & candidate_df["process_state__effusionDemand_W"].notna()
    & candidate_df["process_state__effusionDemand_W"].gt(0)
)
candidate_df.loc[valid_flux_mask, "effusion_flux_ratio_same_tick"] = (
    candidate_df.loc[valid_flux_mask, "effusion__heatInput_w"]
    / candidate_df.loc[valid_flux_mask, "process_state__effusionDemand_W"]
)

candidate_df["effusion_temp_ratio_same_tick"] = np.nan
valid_temp_mask = (
    candidate_df["effusion__act_temp_K"].notna()
    & candidate_df["effusion__target_temp_K"].notna()
    & candidate_df["effusion__target_temp_K"].gt(0)
)
candidate_df.loc[valid_temp_mask, "effusion_temp_ratio_same_tick"] = (
    candidate_df.loc[valid_temp_mask, "effusion__act_temp_K"]
    / candidate_df.loc[valid_temp_mask, "effusion__target_temp_K"]
)

cap_one_effusion_rows = []
cap_one_bad_cap_rows = []
cap_one_missing_threshold_rows = []
cap_one_runs_checked = 0

for run_id, run_df in candidate_df.groupby("run_id", dropna=False):
    run_df = run_df.sort_values("canonical_tick").copy()
    run_meta = run_df.iloc[0]

    underflux_cap_raw = run_meta.get("effusion_underflux_streak_cap", np.nan)
    undertemp_cap_raw = run_meta.get("effusion_undertemp_streak_cap", np.nan)

    underflux_cap = valid_integer_cap(underflux_cap_raw)
    undertemp_cap = valid_integer_cap(undertemp_cap_raw)

    min_flux_fraction = pd.to_numeric(
        run_meta.get("effusion_min_flux_fraction", np.nan),
        errors="coerce",
    )
    temp_tolerance_fraction = pd.to_numeric(
        run_meta.get("effusion_temp_tolerance_fraction", np.nan),
        errors="coerce",
    )

    # keep an audit trail for invalid caps
    if pd.isna(underflux_cap):
        cap_one_bad_cap_rows.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "underflux",
                "cap_raw_value": underflux_cap_raw,
            }
        )

    if pd.isna(undertemp_cap):
        cap_one_bad_cap_rows.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "undertemp",
                "cap_raw_value": undertemp_cap_raw,
            }
        )

    # only the special cap one runs are inspected here
    if (underflux_cap != 1) and (undertemp_cap != 1):
        continue

    cap_one_runs_checked += 1

    # underflux cap one cause
    if underflux_cap == 1:
        if pd.isna(min_flux_fraction):
            cap_one_missing_threshold_rows.append(
                {
                    "run_id": run_id,
                    "config_label": run_meta["config_label"],
                    "job_name": run_meta["job_name"],
                    "threshold_type": "effusion_min_flux_fraction",
                    "threshold_value": min_flux_fraction,
                }
            )
        else:
            underflux_hit_df = run_df.loc[
                run_df["same_tick_live_flag"].eq(1)
                & run_df["effusion_flux_ratio_same_tick"].notna()
                & run_df["effusion_flux_ratio_same_tick"].lt(min_flux_fraction)
            ].copy()

            if not underflux_hit_df.empty:
                first_row = underflux_hit_df.iloc[0]

                cap_one_effusion_rows.append(
                    {
                        "failure_reason": "effusion_underflux_cap_one_same_tick",
                        "run_id": run_id,
                        "config_label": run_meta["config_label"],
                        "job_name": run_meta["job_name"],
                        "failure_tick": first_row["canonical_tick"],
                        "same_tick_live_flag": first_row["same_tick_live_flag"],
                        "effusion_underflux_cap": underflux_cap,
                        "effusion_undertemp_cap": undertemp_cap,
                        "effusion_min_flux_fraction_cfg": min_flux_fraction,
                        "effusion_temp_tolerance_fraction_cfg": temp_tolerance_fraction,
                        "effusion__heatInput_w": first_row.get("effusion__heatInput_w", np.nan),
                        "process_state__effusionDemand_W": first_row.get("process_state__effusionDemand_W", np.nan),
                        "effusion_flux_ratio_same_tick": first_row.get("effusion_flux_ratio_same_tick", np.nan),
                        "effusion__act_temp_K": first_row.get("effusion__act_temp_K", np.nan),
                        "effusion__target_temp_K": first_row.get("effusion__target_temp_K", np.nan),
                        "effusion_temp_ratio_same_tick": first_row.get("effusion_temp_ratio_same_tick", np.nan),
                        "effusion__underflux_streak": first_row.get("effusion__underflux_streak", np.nan),
                        "effusion__temp_miss_streak": first_row.get("effusion__temp_miss_streak", np.nan),
                        "effusion_raw_tick": first_row.get("effusion__tick", np.nan),
                        "process_state_raw_tick": first_row.get("process_state__tick", np.nan),
                        "simulation_engine_raw_tick": first_row.get("simulation_engine__tick", np.nan),
                        "process_state__mbe_flag": first_row.get("process_state__mbe_flag", np.nan),
                        "process_state__deposition_requested": first_row.get("process_state__deposition_requested", np.nan),
                        "process_state__phase_ready_for_execution": first_row.get("process_state__phase_ready_for_execution", np.nan),
                        "simulation_engine__job_failed": first_row.get("simulation_engine__job_failed", np.nan),
                        "simulation_engine__total_power_requested_W": first_row.get("simulation_engine__total_power_requested_W", np.nan),
                        "simulation_engine__total_power_granted_W": first_row.get("simulation_engine__total_power_granted_W", np.nan),
                    }
                )

    # undertemp cap one cause
    if undertemp_cap == 1:
        if pd.isna(temp_tolerance_fraction):
            cap_one_missing_threshold_rows.append(
                {
                    "run_id": run_id,
                    "config_label": run_meta["config_label"],
                    "job_name": run_meta["job_name"],
                    "threshold_type": "effusion_temp_tolerance_fraction",
                    "threshold_value": temp_tolerance_fraction,
                }
            )
        else:
            undertemp_hit_df = run_df.loc[
                run_df["same_tick_live_flag"].eq(1)
                & run_df["effusion_temp_ratio_same_tick"].notna()
                & run_df["effusion_temp_ratio_same_tick"].lt(temp_tolerance_fraction)
            ].copy()

            if not undertemp_hit_df.empty:
                first_row = undertemp_hit_df.iloc[0]

                cap_one_effusion_rows.append(
                    {
                        "failure_reason": "effusion_undertemp_cap_one_same_tick",
                        "run_id": run_id,
                        "config_label": run_meta["config_label"],
                        "job_name": run_meta["job_name"],
                        "failure_tick": first_row["canonical_tick"],
                        "same_tick_live_flag": first_row["same_tick_live_flag"],
                        "effusion_underflux_cap": underflux_cap,
                        "effusion_undertemp_cap": undertemp_cap,
                        "effusion_min_flux_fraction_cfg": min_flux_fraction,
                        "effusion_temp_tolerance_fraction_cfg": temp_tolerance_fraction,
                        "effusion__heatInput_w": first_row.get("effusion__heatInput_w", np.nan),
                        "process_state__effusionDemand_W": first_row.get("process_state__effusionDemand_W", np.nan),
                        "effusion_flux_ratio_same_tick": first_row.get("effusion_flux_ratio_same_tick", np.nan),
                        "effusion__act_temp_K": first_row.get("effusion__act_temp_K", np.nan),
                        "effusion__target_temp_K": first_row.get("effusion__target_temp_K", np.nan),
                        "effusion_temp_ratio_same_tick": first_row.get("effusion_temp_ratio_same_tick", np.nan),
                        "effusion__underflux_streak": first_row.get("effusion__underflux_streak", np.nan),
                        "effusion__temp_miss_streak": first_row.get("effusion__temp_miss_streak", np.nan),
                        "effusion_raw_tick": first_row.get("effusion__tick", np.nan),
                        "process_state_raw_tick": first_row.get("process_state__tick", np.nan),
                        "simulation_engine_raw_tick": first_row.get("simulation_engine__tick", np.nan),
                        "process_state__mbe_flag": first_row.get("process_state__mbe_flag", np.nan),
                        "process_state__deposition_requested": first_row.get("process_state__deposition_requested", np.nan),
                        "process_state__phase_ready_for_execution": first_row.get("process_state__phase_ready_for_execution", np.nan),
                        "simulation_engine__job_failed": first_row.get("simulation_engine__job_failed", np.nan),
                        "simulation_engine__total_power_requested_W": first_row.get("simulation_engine__total_power_requested_W", np.nan),
                        "simulation_engine__total_power_granted_W": first_row.get("simulation_engine__total_power_granted_W", np.nan),
                    }
                )

if len(cap_one_effusion_rows) == 0:
    effusion_cap_one_hits_df = pd.DataFrame(
        columns=[
            "failure_reason",
            "run_id",
            "config_label",
            "job_name",
            "failure_tick",
            "same_tick_live_flag",
            "effusion_underflux_cap",
            "effusion_undertemp_cap",
            "effusion_min_flux_fraction_cfg",
            "effusion_temp_tolerance_fraction_cfg",
            "effusion__heatInput_w",
            "process_state__effusionDemand_W",
            "effusion_flux_ratio_same_tick",
            "effusion__act_temp_K",
            "effusion__target_temp_K",
            "effusion_temp_ratio_same_tick",
            "effusion__underflux_streak",
            "effusion__temp_miss_streak",
            "effusion_raw_tick",
            "process_state_raw_tick",
            "simulation_engine_raw_tick",
            "process_state__mbe_flag",
            "process_state__deposition_requested",
            "process_state__phase_ready_for_execution",
            "simulation_engine__job_failed",
            "simulation_engine__total_power_requested_W",
            "simulation_engine__total_power_granted_W",
        ]
    )
else:
    effusion_cap_one_hits_df = pd.DataFrame(cap_one_effusion_rows).sort_values(
        ["config_label", "job_name", "failure_tick", "failure_reason"],
        na_position="last",
    ).reset_index(drop=True)

# keep only the first cap one failure per run for the funnel
if effusion_cap_one_hits_df.empty:
    effusion_cap_one_first_failures_df = effusion_cap_one_hits_df.copy()
else:
    effusion_cap_one_first_failures_df = (
        effusion_cap_one_hits_df
        .sort_values(["run_id", "failure_tick", "failure_reason"], na_position="last")
        .drop_duplicates(subset=["run_id"], keep="first")
        .reset_index(drop=True)
    )

# update the run level funnel table
if not effusion_cap_one_first_failures_df.empty:
    for _, fail_row in effusion_cap_one_first_failures_df.iterrows():
        run_id = fail_row["run_id"]
        failure_reason = fail_row["failure_reason"]
        failure_tick = fail_row["failure_tick"]

        mask = run_funnel_df["run_id"].astype(str).eq(str(run_id)) & run_funnel_df["funnel_failed"].eq(0)

        run_funnel_df.loc[mask, "funnel_failed"] = 1
        run_funnel_df.loc[mask, "funnel_failure_reason"] = failure_reason
        run_funnel_df.loc[mask, "funnel_failure_tick"] = failure_tick
        run_funnel_df.loc[mask, "funnel_failure_step"] = "effusion_cap_one_same_tick"
        run_funnel_df.loc[mask, "failed_effusion_cap_one_hit"] = 1

        if failure_reason == "effusion_underflux_cap_one_same_tick":
            run_funnel_df.loc[mask, "failed_effusion_underflux_cap_one_hit"] = 1

        if failure_reason == "effusion_undertemp_cap_one_same_tick":
            run_funnel_df.loc[mask, "failed_effusion_undertemp_cap_one_hit"] = 1

# push the updated run level funnel status back onto the full per tick dataframe
run_failed_map = run_funnel_df.set_index("run_id")["funnel_failed"].to_dict()
run_failure_reason_map = run_funnel_df.set_index("run_id")["funnel_failure_reason"].to_dict()
run_failure_tick_map = run_funnel_df.set_index("run_id")["funnel_failure_tick"].to_dict()
run_failure_step_map = run_funnel_df.set_index("run_id")["funnel_failure_step"].to_dict()

run_failed_effusion_cap_one_map = run_funnel_df.set_index("run_id")["failed_effusion_cap_one_hit"].to_dict()
run_failed_effusion_underflux_cap_one_map = run_funnel_df.set_index("run_id")["failed_effusion_underflux_cap_one_hit"].to_dict()
run_failed_effusion_undertemp_cap_one_map = run_funnel_df.set_index("run_id")["failed_effusion_undertemp_cap_one_hit"].to_dict()

all_runs_tick_df["funnel_failed"] = (
    all_runs_tick_df["run_id"].map(run_failed_map).fillna(0).astype(int)
)
all_runs_tick_df["funnel_failure_reason"] = all_runs_tick_df["run_id"].map(run_failure_reason_map)
all_runs_tick_df["funnel_failure_tick"] = all_runs_tick_df["run_id"].map(run_failure_tick_map)
all_runs_tick_df["funnel_failure_step"] = all_runs_tick_df["run_id"].map(run_failure_step_map)
all_runs_tick_df["failed_effusion_cap_one_hit"] = (
    all_runs_tick_df["run_id"].map(run_failed_effusion_cap_one_map).fillna(0).astype(int)
)
all_runs_tick_df["failed_effusion_underflux_cap_one_hit"] = (
    all_runs_tick_df["run_id"].map(run_failed_effusion_underflux_cap_one_map).fillna(0).astype(int)
)
all_runs_tick_df["failed_effusion_undertemp_cap_one_hit"] = (
    all_runs_tick_df["run_id"].map(run_failed_effusion_undertemp_cap_one_map).fillna(0).astype(int)
)
all_runs_tick_df["funnel_is_failure_tick"] = (
    all_runs_tick_df["funnel_failed"].eq(1)
    & pd.to_numeric(all_runs_tick_df["canonical_tick"], errors="coerce").eq(
        pd.to_numeric(all_runs_tick_df["funnel_failure_tick"], errors="coerce")
    )
).astype(int)

print("funnel step name", "effusion_cap_one_same_tick")
print("runs eligible at step start", len(eligible_run_ids))
print("runs with cap one settings checked", cap_one_runs_checked)
print("runs newly failed in this step", len(effusion_cap_one_first_failures_df))
print("runs still alive after this step", int((run_funnel_df["funnel_failed"] == 0).sum()))

if len(cap_one_bad_cap_rows) > 0:
    print("these runs had cap values that were not valid whole numbers")
    display(
        pd.DataFrame(cap_one_bad_cap_rows)
        .drop_duplicates(subset=["config_label", "cap_type"])
        .sort_values(["config_label", "cap_type"])
        .reset_index(drop=True)
    )

if len(cap_one_missing_threshold_rows) > 0:
    print("these cap one causes were skipped because the required threshold value was missing")
    display(
        pd.DataFrame(cap_one_missing_threshold_rows)
        .drop_duplicates(subset=["config_label", "threshold_type"])
        .sort_values(["config_label", "threshold_type"])
        .reset_index(drop=True)
    )

if effusion_cap_one_hits_df.empty:
    print("no same tick effusion cap one rows were found among the currently alive runs")
else:
    print("exact merged tick rows where a same tick effusion cap one failure was first observed")
    print("rows shown", len(effusion_cap_one_hits_df))
    display(effusion_cap_one_hits_df)

    print("first cap one effusion failure stamped into the funnel for each affected run")
    display(effusion_cap_one_first_failures_df)

print("current run level funnel status")
display(
    run_funnel_df.sort_values(["config_label", "job_name"]).reset_index(drop=True)
)

print("sample of the full per tick dataframe after funnel annotation")
show_cols = [
    "run_id",
    "config_label",
    "job_name",
    "canonical_tick",
    "funnel_failed",
    "funnel_failure_reason",
    "funnel_failure_tick",
    "funnel_failure_step",
    "funnel_is_failure_tick",
    "failed_effusion_cap_one_hit",
    "failed_effusion_underflux_cap_one_hit",
    "failed_effusion_undertemp_cap_one_hit",
]
show_cols = [col for col in show_cols if col in all_runs_tick_df.columns]

display(
    all_runs_tick_df[show_cols]
    .sort_values(["config_label", "job_name", "canonical_tick"])
    .head(30)
)

# funnel step 3
# mark runs that fail because the raw substrate undertemp streak hits the config cap
#
# important logic
# this step only checks the substrate failure streak
# it does not rebuild features
# it does not inspect effusion causes
#
# simulator semantics
# the substrate miss streak increments only when the substrate is below the lower ready bound
# that lower bound is target minus ready band
# the job fails when that streak reaches the substrate fail limit
#
# funnel rule
# only runs that are still alive are inspected in this step
# once a run is marked as failed here it should not be inspected by later funnel steps

# add the run level columns for this funnel step if they do not exist yet
substrate_default_columns = {
    "failed_substrate_cap_hit": 0,
}

for col, default_value in substrate_default_columns.items():
    if col not in run_funnel_df.columns:
        run_funnel_df[col] = default_value

run_funnel_df["failed_substrate_cap_hit"] = pd.to_numeric(
    run_funnel_df["failed_substrate_cap_hit"],
    errors="coerce",
).fillna(0).astype(int)

eligible_run_ids = set(
    run_funnel_df.loc[run_funnel_df["funnel_failed"] == 0, "run_id"].astype(str).tolist()
)

required_cols = [
    "run_id",
    "config_label",
    "job_name",
    "canonical_tick",
    "substrate_fail_limit_ticks",
    "ready_band_k",
    "substrate__streak",
    "substrate__T_sub_K",
    "substrate__T_target_K",
]

missing_required_cols = [col for col in required_cols if col not in all_runs_tick_df.columns]
if missing_required_cols:
    raise ValueError(f"missing required columns in all_runs_tick_df {missing_required_cols}")

candidate_df = all_runs_tick_df.loc[
    all_runs_tick_df["run_id"].astype(str).isin(eligible_run_ids)
].copy()

numeric_cols = [
    "canonical_tick",
    "substrate_fail_limit_ticks",
    "ready_band_k",
    "substrate__streak",
    "substrate__T_sub_K",
    "substrate__T_target_K",
]

optional_numeric_cols = [
    "substrate__tick",
    "process_state__tick",
    "simulation_engine__tick",
    "substrate__failed",
    "substrate__job_active",
    "substrate__substrate_control_on",
]

for col in numeric_cols:
    candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")

for col in optional_numeric_cols:
    if col in candidate_df.columns:
        candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")

# audit columns
candidate_df["substrate_temp_error_k"] = (
    candidate_df["substrate__T_sub_K"] - candidate_df["substrate__T_target_K"]
)
candidate_df["substrate_lower_ready_bound_k"] = (
    candidate_df["substrate__T_target_K"] - candidate_df["ready_band_k"]
)
candidate_df["substrate_below_lower_band_flag"] = (
    candidate_df["substrate__T_sub_K"].notna()
    & candidate_df["substrate__T_target_K"].notna()
    & candidate_df["ready_band_k"].notna()
    & candidate_df["substrate__T_sub_K"].lt(candidate_df["substrate_lower_ready_bound_k"])
).astype(int)

substrate_cap_hit_rows = []
substrate_bad_cap_rows = []
substrate_runs_checked = 0

for run_id, run_df in candidate_df.groupby("run_id", dropna=False):
    run_df = run_df.sort_values("canonical_tick").copy()
    run_meta = run_df.iloc[0]

    substrate_cap_raw = run_meta.get("substrate_fail_limit_ticks", np.nan)
    substrate_cap = valid_integer_cap(substrate_cap_raw)
    ready_band_value = pd.to_numeric(run_meta.get("ready_band_k", np.nan), errors="coerce")

    if pd.isna(substrate_cap):
        substrate_bad_cap_rows.append(
            {
                "run_id": run_id,
                "config_label": run_meta["config_label"],
                "job_name": run_meta["job_name"],
                "cap_type": "substrate_fail_limit_ticks",
                "cap_raw_value": substrate_cap_raw,
            }
        )
        continue

    substrate_runs_checked += 1

    # the true failure tick is the first tick where the raw substrate streak equals the cap
    exact_hit_df = run_df.loc[run_df["substrate__streak"].eq(substrate_cap)].copy()

    # defensive fallback in case an exact equality row is missing in a rare run
    ge_hit_df = run_df.loc[run_df["substrate__streak"].ge(substrate_cap)].copy()

    if not exact_hit_df.empty:
        first_row = exact_hit_df.iloc[0]
        match_type = "exact_cap_tick"
    elif not ge_hit_df.empty:
        first_row = ge_hit_df.iloc[0]
        match_type = "ge_fallback"
    else:
        continue

    substrate_cap_hit_rows.append(
        {
            "failure_reason": "substrate_undertemp_cap_hit",
            "run_id": run_id,
            "config_label": run_meta["config_label"],
            "job_name": run_meta["job_name"],
            "failure_tick": first_row["canonical_tick"],
            "substrate_fail_limit_ticks": substrate_cap,
            "ready_band_k_cfg": ready_band_value,
            "substrate_streak_value": first_row.get("substrate__streak", np.nan),
            "substrate_actual_temp_k": first_row.get("substrate__T_sub_K", np.nan),
            "substrate_target_temp_k": first_row.get("substrate__T_target_K", np.nan),
            "substrate_temp_error_k": first_row.get("substrate_temp_error_k", np.nan),
            "substrate_lower_ready_bound_k": first_row.get("substrate_lower_ready_bound_k", np.nan),
            "substrate_below_lower_band_flag": first_row.get("substrate_below_lower_band_flag", np.nan),
            "substrate_raw_tick": first_row.get("substrate__tick", np.nan),
            "process_state_raw_tick": first_row.get("process_state__tick", np.nan),
            "simulation_engine_raw_tick": first_row.get("simulation_engine__tick", np.nan),
            "substrate_failed_logged": first_row.get("substrate__failed", np.nan),
            "substrate_job_active": first_row.get("substrate__job_active", np.nan),
            "substrate_control_on_logged": first_row.get("substrate__substrate_control_on", np.nan),
            "cap_hit_match_type": match_type,
        }
    )

if len(substrate_cap_hit_rows) == 0:
    substrate_cap_hits_df = pd.DataFrame(
        columns=[
            "failure_reason",
            "run_id",
            "config_label",
            "job_name",
            "failure_tick",
            "substrate_fail_limit_ticks",
            "ready_band_k_cfg",
            "substrate_streak_value",
            "substrate_actual_temp_k",
            "substrate_target_temp_k",
            "substrate_temp_error_k",
            "substrate_lower_ready_bound_k",
            "substrate_below_lower_band_flag",
            "substrate_raw_tick",
            "process_state_raw_tick",
            "simulation_engine_raw_tick",
            "substrate_failed_logged",
            "substrate_job_active",
            "substrate_control_on_logged",
            "cap_hit_match_type",
        ]
    )
else:
    substrate_cap_hits_df = pd.DataFrame(substrate_cap_hit_rows).sort_values(
        ["config_label", "job_name", "failure_tick"],
        na_position="last",
    ).reset_index(drop=True)

# one substrate failure stamp per run for the funnel
if substrate_cap_hits_df.empty:
    substrate_first_failures_df = substrate_cap_hits_df.copy()
else:
    substrate_first_failures_df = (
        substrate_cap_hits_df
        .sort_values(["run_id", "failure_tick"], na_position="last")
        .drop_duplicates(subset=["run_id"], keep="first")
        .reset_index(drop=True)
    )

# update the run level funnel table
if not substrate_first_failures_df.empty:
    for _, fail_row in substrate_first_failures_df.iterrows():
        run_id = fail_row["run_id"]
        failure_reason = fail_row["failure_reason"]
        failure_tick = fail_row["failure_tick"]

        mask = run_funnel_df["run_id"].astype(str).eq(str(run_id)) & run_funnel_df["funnel_failed"].eq(0)

        run_funnel_df.loc[mask, "funnel_failed"] = 1
        run_funnel_df.loc[mask, "funnel_failure_reason"] = failure_reason
        run_funnel_df.loc[mask, "funnel_failure_tick"] = failure_tick
        run_funnel_df.loc[mask, "funnel_failure_step"] = "substrate_cap_hit"
        run_funnel_df.loc[mask, "failed_substrate_cap_hit"] = 1

# push the updated run level funnel status back onto the full per tick dataframe
run_failed_map = run_funnel_df.set_index("run_id")["funnel_failed"].to_dict()
run_failure_reason_map = run_funnel_df.set_index("run_id")["funnel_failure_reason"].to_dict()
run_failure_tick_map = run_funnel_df.set_index("run_id")["funnel_failure_tick"].to_dict()
run_failure_step_map = run_funnel_df.set_index("run_id")["funnel_failure_step"].to_dict()
run_failed_substrate_cap_map = run_funnel_df.set_index("run_id")["failed_substrate_cap_hit"].to_dict()

all_runs_tick_df["funnel_failed"] = (
    all_runs_tick_df["run_id"].map(run_failed_map).fillna(0).astype(int)
)
all_runs_tick_df["funnel_failure_reason"] = all_runs_tick_df["run_id"].map(run_failure_reason_map)
all_runs_tick_df["funnel_failure_tick"] = all_runs_tick_df["run_id"].map(run_failure_tick_map)
all_runs_tick_df["funnel_failure_step"] = all_runs_tick_df["run_id"].map(run_failure_step_map)
all_runs_tick_df["failed_substrate_cap_hit"] = (
    all_runs_tick_df["run_id"].map(run_failed_substrate_cap_map).fillna(0).astype(int)
)
all_runs_tick_df["funnel_is_failure_tick"] = (
    all_runs_tick_df["funnel_failed"].eq(1)
    & pd.to_numeric(all_runs_tick_df["canonical_tick"], errors="coerce").eq(
        pd.to_numeric(all_runs_tick_df["funnel_failure_tick"], errors="coerce")
    )
).astype(int)

# also annotate the engineered feature table when it exists
if "all_runs_features_df" in globals() and isinstance(all_runs_features_df, pd.DataFrame) and not all_runs_features_df.empty:
    all_runs_features_df["funnel_failed"] = (
        all_runs_features_df["run_id"].map(run_failed_map).fillna(0).astype(int)
    )
    all_runs_features_df["funnel_failure_reason"] = all_runs_features_df["run_id"].map(run_failure_reason_map)
    all_runs_features_df["funnel_failure_tick"] = all_runs_features_df["run_id"].map(run_failure_tick_map)
    all_runs_features_df["funnel_failure_step"] = all_runs_features_df["run_id"].map(run_failure_step_map)
    all_runs_features_df["failed_substrate_cap_hit"] = (
        all_runs_features_df["run_id"].map(run_failed_substrate_cap_map).fillna(0).astype(int)
    )
    all_runs_features_df["funnel_is_failure_tick"] = (
        all_runs_features_df["funnel_failed"].eq(1)
        & pd.to_numeric(all_runs_features_df["canonical_tick"], errors="coerce").eq(
            pd.to_numeric(all_runs_features_df["funnel_failure_tick"], errors="coerce")
        )
    ).astype(int)

print("funnel step name", "substrate_cap_hit")
print("runs eligible at step start", len(eligible_run_ids))
print("runs with valid substrate cap checked", substrate_runs_checked)
print("runs newly failed in this step", len(substrate_first_failures_df))
print("runs still alive after this step", int((run_funnel_df["funnel_failed"] == 0).sum()))

if len(substrate_bad_cap_rows) > 0:
    print("these runs had substrate fail limit values that were not valid whole numbers")
    display(
        pd.DataFrame(substrate_bad_cap_rows)
        .drop_duplicates(subset=["config_label", "cap_type"])
        .sort_values(["config_label", "cap_type"])
        .reset_index(drop=True)
    )

if substrate_cap_hits_df.empty:
    print("no raw substrate streak rows hit the substrate fail limit among the currently alive runs")
else:
    print("exact merged tick rows where the raw substrate undertemp streak first hit the fail limit")
    print("rows shown", len(substrate_cap_hits_df))
    display(substrate_cap_hits_df)

    print("first substrate failure stamped into the funnel for each affected run")
    display(substrate_first_failures_df)

print("current run level funnel status")
display(
    run_funnel_df.sort_values(["config_label", "job_name"]).reset_index(drop=True)
)

print("sample of the full per tick dataframe after funnel annotation")
show_cols = [
    "run_id",
    "config_label",
    "job_name",
    "canonical_tick",
    "funnel_failed",
    "funnel_failure_reason",
    "funnel_failure_tick",
    "funnel_failure_step",
    "funnel_is_failure_tick",
    "failed_substrate_cap_hit",
]
show_cols = [col for col in show_cols if col in all_runs_tick_df.columns]

display(
    all_runs_tick_df[show_cols]
    .sort_values(["config_label", "job_name", "canonical_tick"])
    .head(30)
)

# stall detection with easy threshold controls
#
# default interpretation
# one tick is one minute
# below 30 minutes does not count
# from 30 minutes up to 59 minutes it is noted but not failed
# at 60 minutes or more it becomes a failure
# and the run is cut at the 60th stalled minute
#
# edit these two values only when a different stall policy is needed
STALL_NOTE_MINUTES = 30
STALL_FAIL_MINUTES = 60

if STALL_NOTE_MINUTES < 1:
    raise ValueError("STALL_NOTE_MINUTES must be at least 1")

if STALL_FAIL_MINUTES < STALL_NOTE_MINUTES:
    raise ValueError("STALL_FAIL_MINUTES must be greater than or equal to STALL_NOTE_MINUTES")

if "all_runs_features_df" not in globals() or all_runs_features_df.empty:
    raise ValueError("all_runs_features_df is required before running the stall step")

def numeric_series_or_default(df: pd.DataFrame, col_name: str, default_value=np.nan) -> pd.Series:
    """
    read one numeric series when it exists
    otherwise return one fallback series of the same length
    """
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce")
    return pd.Series(default_value, index=df.index, dtype=float)

def object_series_or_default(df: pd.DataFrame, col_name: str, default_value=pd.NA) -> pd.Series:
    """
    read one object series when it exists
    otherwise return one fallback series of the same length
    """
    if col_name in df.columns:
        return df[col_name].astype("object")
    return pd.Series([default_value] * len(df), index=df.index, dtype="object")

def first_tick_from_flag(df: pd.DataFrame, flag_col: str):
    """
    get the first canonical tick where one flag becomes one
    """
    if flag_col not in df.columns:
        return np.nan

    ticks = pd.to_numeric(df["canonical_tick"], errors="coerce")
    flag = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).eq(1)

    if not flag.any():
        return np.nan

    return ticks.loc[flag].iloc[0]

def build_logged_failure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    build simple logged failure columns from the simulation log
    """
    out = df.copy()

    if "failure_logged_simulation" in out.columns:
        failure_logged = pd.to_numeric(out["failure_logged_simulation"], errors="coerce").fillna(0)
    elif "simulation_job_failed" in out.columns:
        failure_logged = pd.to_numeric(out["simulation_job_failed"], errors="coerce").fillna(0)
    elif "simulation_engine__job_failed" in out.columns:
        failure_logged = pd.to_numeric(out["simulation_engine__job_failed"], errors="coerce").fillna(0)
    else:
        failure_logged = pd.Series(0, index=out.index, dtype=float)

    out["failure_logged_simulation"] = failure_logged.astype(int)
    out["first_logged_failure_tick"] = first_tick_from_flag(out, "failure_logged_simulation")

    return out

def initialize_stall_columns_for_unchecked_run(df: pd.DataFrame, note_minutes: int, fail_minutes: int) -> pd.DataFrame:
    """
    add empty stall columns for a run that is not being checked in this step
    """
    out = build_logged_failure_columns(df.copy())

    out["stall_policy_note_minutes"] = note_minutes
    out["stall_policy_fail_minutes"] = fail_minutes
    out["stall_checked_in_step_flag"] = 0

    out["stall_candidate_state_flag"] = 0
    out["stall_raw_flag"] = 0
    out["stall_streak_ticks"] = 0

    out["stall_note_30_to_59_flag"] = 0
    out["stall_failure_60plus_flag"] = 0

    out["first_stall_note_tick"] = np.nan
    out["first_stall_failure_tick"] = np.nan
    out["stall_cutoff_tick"] = np.nan

    out["run_has_stall_note_30_to_59"] = 0
    out["run_has_stall_failure_60plus"] = 0

    out["keep_through_stall_cutoff_flag"] = 1

    return out

def build_stall_columns_one_run(df: pd.DataFrame, note_minutes: int = 30, fail_minutes: int = 60) -> pd.DataFrame:
    """
    build stall columns for one run

    stall candidate states
    warmup
    cooldown
    thermal prep

    stall definition
    the same run must stay on consecutive ticks
    the same subjob must remain active
    the same phase must remain active
    phase progress must stay frozen
    live progress must stay frozen
    and the simulator must not already have logged a failure on that row

    threshold logic
    below note threshold nothing is counted
    from note threshold up to one less than fail threshold it is only noted
    at fail threshold or above it is a stall failure
    """
    out = build_logged_failure_columns(df.sort_values("canonical_tick").reset_index(drop=True).copy())

    out["stall_policy_note_minutes"] = note_minutes
    out["stall_policy_fail_minutes"] = fail_minutes
    out["stall_checked_in_step_flag"] = 1

    warmup_active = numeric_series_or_default(out, "schedule_state__warmup_active", default_value=0).fillna(0)
    cooldown_active = numeric_series_or_default(out, "schedule_state__cooldown_active", default_value=0).fillna(0)
    thermal_prep_active = numeric_series_or_default(out, "schedule_state__thermal_prep_active", default_value=0).fillna(0)

    out["stall_candidate_state_flag"] = (
        warmup_active.eq(1)
        | cooldown_active.eq(1)
        | thermal_prep_active.eq(1)
    ).astype(int)

    phase_ticks_completed = numeric_series_or_default(
        out,
        "schedule_state__phase_ticks_completed",
        default_value=np.nan,
    )

    live_ticks_completed = numeric_series_or_default(
        out,
        "schedule_state__live_ticks_completed",
        default_value=np.nan,
    )

    job_index = numeric_series_or_default(out, "actual_subjob_index", default_value=np.nan)
    if job_index.isna().all():
        job_index = numeric_series_or_default(out, "process_controlling_job_index", default_value=np.nan)
    if job_index.isna().all():
        job_index = numeric_series_or_default(out, "schedule_controlling_job_index", default_value=np.nan)

    phase_name = object_series_or_default(out, "phase_name_resolved", default_value=pd.NA)
    canonical_tick = pd.to_numeric(out["canonical_tick"], errors="coerce")

    continuous_tick_flag = canonical_tick.diff().eq(1).fillna(False)

    same_job_flag = (
        job_index.notna()
        & job_index.shift(1).notna()
        & job_index.eq(job_index.shift(1))
    )

    same_phase_flag = (
        phase_name.notna()
        & phase_name.shift(1).notna()
        & phase_name.eq(phase_name.shift(1))
    )

    same_phase_progress_flag = (
        phase_ticks_completed.notna()
        & phase_ticks_completed.shift(1).notna()
        & phase_ticks_completed.eq(phase_ticks_completed.shift(1))
    )

    same_live_progress_flag = (
        live_ticks_completed.notna()
        & live_ticks_completed.shift(1).notna()
        & live_ticks_completed.eq(live_ticks_completed.shift(1))
    )

    out["stall_raw_flag"] = (
        continuous_tick_flag
        & same_job_flag
        & same_phase_flag
        & same_phase_progress_flag
        & same_live_progress_flag
        & pd.to_numeric(out["stall_candidate_state_flag"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(out["failure_logged_simulation"], errors="coerce").fillna(0).eq(0)
    ).astype(int)

    out["stall_streak_ticks"] = consecutive_count_from_flag(out["stall_raw_flag"])

    # this is the note only range
    # it covers note threshold up to one less than fail threshold
    out["stall_note_30_to_59_flag"] = (
        pd.to_numeric(out["stall_streak_ticks"], errors="coerce").ge(note_minutes)
        & pd.to_numeric(out["stall_streak_ticks"], errors="coerce").lt(fail_minutes)
    ).astype(int)

    # this is the actual stall failure
    out["stall_failure_60plus_flag"] = (
        pd.to_numeric(out["stall_streak_ticks"], errors="coerce").ge(fail_minutes)
    ).astype(int)

    first_stall_note_tick = first_tick_from_flag(out, "stall_note_30_to_59_flag")
    first_stall_failure_tick = first_tick_from_flag(out, "stall_failure_60plus_flag")

    out["first_stall_note_tick"] = first_stall_note_tick
    out["first_stall_failure_tick"] = first_stall_failure_tick
    out["stall_cutoff_tick"] = first_stall_failure_tick

    run_has_stall_failure_60plus = int(pd.notna(first_stall_failure_tick))
    run_has_stall_note_30_to_59 = int((pd.notna(first_stall_note_tick)) and (run_has_stall_failure_60plus == 0))

    out["run_has_stall_note_30_to_59"] = run_has_stall_note_30_to_59
    out["run_has_stall_failure_60plus"] = run_has_stall_failure_60plus

    if run_has_stall_failure_60plus == 1:
        out["keep_through_stall_cutoff_flag"] = (
            pd.to_numeric(out["canonical_tick"], errors="coerce").le(first_stall_failure_tick)
        ).astype(int)
    else:
        out["keep_through_stall_cutoff_flag"] = 1

    return out

# stall step uses only runs still alive in the current funnel
if "run_funnel_df" in globals():
    stall_eligible_run_ids = set(
        run_funnel_df.loc[run_funnel_df["funnel_failed"] == 0, "run_id"].astype(str).tolist()
    )
else:
    stall_eligible_run_ids = set(all_runs_features_df["run_id"].dropna().astype(str).tolist())

stall_feature_tables = []
stall_summary_rows = []

for run_id, run_df in all_runs_features_df.groupby("run_id", sort=False, dropna=False):
    run_id_str = str(run_id)
    run_df = run_df.sort_values("canonical_tick").reset_index(drop=True).copy()

    if run_id_str in stall_eligible_run_ids:
        stall_df = build_stall_columns_one_run(
            run_df,
            note_minutes=STALL_NOTE_MINUTES,
            fail_minutes=STALL_FAIL_MINUTES,
        )
    else:
        stall_df = initialize_stall_columns_for_unchecked_run(
            run_df,
            note_minutes=STALL_NOTE_MINUTES,
            fail_minutes=STALL_FAIL_MINUTES,
        )

    stall_feature_tables.append(stall_df)

    first_row = stall_df.iloc[0]

    stall_summary_rows.append(
        {
            "run_id": run_id_str,
            "config_label": first_row.get("config_label", pd.NA),
            "job_name": first_row.get("job_name", pd.NA),
            "stall_checked_in_step_flag": int(first_row.get("stall_checked_in_step_flag", 0)),
            "max_stall_streak_ticks": pd.to_numeric(stall_df["stall_streak_ticks"], errors="coerce").max(),
            "run_has_stall_note_30_to_59": int(first_row.get("run_has_stall_note_30_to_59", 0)),
            "run_has_stall_failure_60plus": int(first_row.get("run_has_stall_failure_60plus", 0)),
            "first_stall_note_tick": first_row.get("first_stall_note_tick", np.nan),
            "first_stall_failure_tick": first_row.get("first_stall_failure_tick", np.nan),
            "stall_cutoff_tick": first_row.get("stall_cutoff_tick", np.nan),
        }
    )

all_runs_stall_df = pd.concat(stall_feature_tables, ignore_index=True, sort=False)
all_runs_stall_df = all_runs_stall_df.sort_values(
    ["config_label", "job_name", "canonical_tick"],
    na_position="last",
).reset_index(drop=True)

stall_run_summary_df = pd.DataFrame(stall_summary_rows).sort_values(
    ["config_label", "job_name"],
    na_position="last",
).reset_index(drop=True)

# update the run level funnel only for actual 60 plus stall failures
if "run_funnel_df" in globals():
    if "failed_stall_60plus" not in run_funnel_df.columns:
        run_funnel_df["failed_stall_60plus"] = 0

    if "noted_stall_30_to_59" not in run_funnel_df.columns:
        run_funnel_df["noted_stall_30_to_59"] = 0

    run_funnel_df["failed_stall_60plus"] = pd.to_numeric(
        run_funnel_df["failed_stall_60plus"],
        errors="coerce",
    ).fillna(0).astype(int)

    run_funnel_df["noted_stall_30_to_59"] = pd.to_numeric(
        run_funnel_df["noted_stall_30_to_59"],
        errors="coerce",
    ).fillna(0).astype(int)

    for _, stall_row in stall_run_summary_df.iterrows():
        run_id = str(stall_row["run_id"])

        note_only_flag = int(stall_row["run_has_stall_note_30_to_59"])
        fail_flag = int(stall_row["run_has_stall_failure_60plus"])
        fail_tick = stall_row["first_stall_failure_tick"]

        mask = run_funnel_df["run_id"].astype(str).eq(run_id)

        if note_only_flag == 1:
            run_funnel_df.loc[mask, "noted_stall_30_to_59"] = 1

        if fail_flag == 1:
            alive_mask = mask & run_funnel_df["funnel_failed"].eq(0)

            run_funnel_df.loc[alive_mask, "funnel_failed"] = 1
            run_funnel_df.loc[alive_mask, "funnel_failure_reason"] = "stall_60plus_cutoff"
            run_funnel_df.loc[alive_mask, "funnel_failure_tick"] = fail_tick
            run_funnel_df.loc[alive_mask, "funnel_failure_step"] = "stall_60plus_cutoff"
            run_funnel_df.loc[alive_mask, "failed_stall_60plus"] = 1

# push stall info back onto the feature dataframe
stall_note_map = stall_run_summary_df.set_index("run_id")["run_has_stall_note_30_to_59"].to_dict()
stall_fail_map = stall_run_summary_df.set_index("run_id")["run_has_stall_failure_60plus"].to_dict()
stall_cutoff_map = stall_run_summary_df.set_index("run_id")["stall_cutoff_tick"].to_dict()

all_runs_stall_df["run_has_stall_note_30_to_59"] = (
    all_runs_stall_df["run_id"].astype(str).map(stall_note_map).fillna(0).astype(int)
)
all_runs_stall_df["run_has_stall_failure_60plus"] = (
    all_runs_stall_df["run_id"].astype(str).map(stall_fail_map).fillna(0).astype(int)
)
all_runs_stall_df["stall_cutoff_tick"] = all_runs_stall_df["run_id"].astype(str).map(stall_cutoff_map)

# update the notebook main feature table
all_runs_features_df = all_runs_stall_df.copy()

# update the raw tick table too when it exists
if "all_runs_tick_df" in globals() and isinstance(all_runs_tick_df, pd.DataFrame) and not all_runs_tick_df.empty:
    all_runs_tick_df["run_has_stall_note_30_to_59"] = (
        all_runs_tick_df["run_id"].astype(str).map(stall_note_map).fillna(0).astype(int)
    )
    all_runs_tick_df["run_has_stall_failure_60plus"] = (
        all_runs_tick_df["run_id"].astype(str).map(stall_fail_map).fillna(0).astype(int)
    )
    all_runs_tick_df["stall_cutoff_tick"] = all_runs_tick_df["run_id"].astype(str).map(stall_cutoff_map)

    all_runs_tick_df["keep_through_stall_cutoff_flag"] = 1
    stall_cutoff_present = pd.to_numeric(all_runs_tick_df["stall_cutoff_tick"], errors="coerce").notna()
    all_runs_tick_df.loc[stall_cutoff_present, "keep_through_stall_cutoff_flag"] = (
        pd.to_numeric(all_runs_tick_df.loc[stall_cutoff_present, "canonical_tick"], errors="coerce")
        .le(pd.to_numeric(all_runs_tick_df.loc[stall_cutoff_present, "stall_cutoff_tick"], errors="coerce"))
    ).astype(int)

# cut the data only for real 60 plus stall failures
all_runs_features_after_stall_cut_df = all_runs_features_df.loc[
    all_runs_features_df["keep_through_stall_cutoff_flag"] == 1
].copy()

if "all_runs_tick_df" in globals() and isinstance(all_runs_tick_df, pd.DataFrame) and not all_runs_tick_df.empty:
    all_runs_tick_after_stall_cut_df = all_runs_tick_df.loc[
        all_runs_tick_df["keep_through_stall_cutoff_flag"] == 1
    ].copy()
else:
    all_runs_tick_after_stall_cut_df = pd.DataFrame()

# save useful outputs
stall_summary_csv_path = OUTPUT_ROOT / "stall_run_summary.csv"
stall_features_parquet_path = OUTPUT_ROOT / "all_runs_features_with_stall.parquet"
stall_features_cut_parquet_path = OUTPUT_ROOT / "all_runs_features_after_stall_cut.parquet"

stall_run_summary_df.to_csv(stall_summary_csv_path, index=False)
all_runs_features_df.to_parquet(stall_features_parquet_path, index=False)
all_runs_features_after_stall_cut_df.to_parquet(stall_features_cut_parquet_path, index=False)

print("stall note threshold minutes", STALL_NOTE_MINUTES)
print("stall failure threshold minutes", STALL_FAIL_MINUTES)
print("runs eligible for stall check", len(stall_eligible_run_ids))
print("runs noted for 30 to 59 minute stall", int(stall_run_summary_df["run_has_stall_note_30_to_59"].sum()))
print("runs failed for 60 plus minute stall", int(stall_run_summary_df["run_has_stall_failure_60plus"].sum()))
print("feature rows before stall cut", len(all_runs_features_df))
print("feature rows after stall cut", len(all_runs_features_after_stall_cut_df))
print("stall summary path", stall_summary_csv_path)
print("stall features path", stall_features_parquet_path)
print("stall cut features path", stall_features_cut_parquet_path)

display(
    stall_run_summary_df[
        [
            "run_id",
            "config_label",
            "job_name",
            "stall_checked_in_step_flag",
            "max_stall_streak_ticks",
            "run_has_stall_note_30_to_59",
            "run_has_stall_failure_60plus",
            "first_stall_note_tick",
            "first_stall_failure_tick",
            "stall_cutoff_tick",
        ]
    ].head(30)
)

sample_stall_cols = [
    "run_id",
    "canonical_tick",
    "actual_subjob_index",
    "phase_name_resolved",
    "failure_logged_simulation",
    "stall_checked_in_step_flag",
    "stall_candidate_state_flag",
    "stall_raw_flag",
    "stall_streak_ticks",
    "stall_note_30_to_59_flag",
    "stall_failure_60plus_flag",
    "first_logged_failure_tick",
    "first_stall_note_tick",
    "first_stall_failure_tick",
    "stall_cutoff_tick",
    "keep_through_stall_cutoff_flag",
    "run_has_stall_note_30_to_59",
    "run_has_stall_failure_60plus",
]
sample_stall_cols = [col for col in sample_stall_cols if col in all_runs_features_df.columns]

display(all_runs_features_df[sample_stall_cols].head(80))

if "run_funnel_df" in globals():
    print("updated run level funnel status after stall step")
    display(
        run_funnel_df.sort_values(["config_label", "job_name"]).reset_index(drop=True)
    )

# failure breakdown across all runs and configs
#
# this cell summarizes how many runs failed due to each failure type
# and also shows the per config breakdown
#
# run meaning
# one run is one job within one config
# so these counts are across all jobs across all configs

if "run_funnel_df" not in globals() or run_funnel_df.empty:
    raise ValueError("run_funnel_df is required before building the failure breakdown")

failure_breakdown_base_df = run_funnel_df.copy()

failure_breakdown_base_df["funnel_failed"] = pd.to_numeric(
    failure_breakdown_base_df["funnel_failed"],
    errors="coerce",
).fillna(0).astype(int)

failed_runs_df = failure_breakdown_base_df.loc[
    failure_breakdown_base_df["funnel_failed"] == 1
].copy()

alive_runs_df = failure_breakdown_base_df.loc[
    failure_breakdown_base_df["funnel_failed"] == 0
].copy()

# overall counts by exact failure reason
if failed_runs_df.empty:
    failure_reason_summary_df = pd.DataFrame(
        columns=["failure_reason", "run_count"]
    )
else:
    failure_reason_summary_df = (
        failed_runs_df["funnel_failure_reason"]
        .fillna("unknown_failure_reason")
        .value_counts(dropna=False)
        .rename_axis("failure_reason")
        .reset_index(name="run_count")
        .sort_values(["run_count", "failure_reason"], ascending=[False, True])
        .reset_index(drop=True)
    )

# overall counts by funnel step
if failed_runs_df.empty:
    failure_step_summary_df = pd.DataFrame(
        columns=["failure_step", "run_count"]
    )
else:
    failure_step_summary_df = (
        failed_runs_df["funnel_failure_step"]
        .fillna("unknown_failure_step")
        .value_counts(dropna=False)
        .rename_axis("failure_step")
        .reset_index(name="run_count")
        .sort_values(["run_count", "failure_step"], ascending=[False, True])
        .reset_index(drop=True)
    )

# counts by boolean failure columns
failure_flag_cols = sorted(
    [
        col
        for col in failure_breakdown_base_df.columns
        if col.startswith("failed_")
    ]
)

failure_flag_summary_rows = []

for col in failure_flag_cols:
    failure_flag_summary_rows.append(
        {
            "failure_flag_column": col,
            "run_count": int(pd.to_numeric(failure_breakdown_base_df[col], errors="coerce").fillna(0).sum()),
        }
    )

failure_flag_summary_df = pd.DataFrame(failure_flag_summary_rows).sort_values(
    ["run_count", "failure_flag_column"],
    ascending=[False, True],
).reset_index(drop=True)

# note only columns that are useful for penalties later
note_flag_cols = sorted(
    [
        col
        for col in failure_breakdown_base_df.columns
        if col.startswith("noted_") or col.startswith("run_has_stall_note")
    ]
)

note_flag_summary_rows = []

for col in note_flag_cols:
    note_flag_summary_rows.append(
        {
            "note_flag_column": col,
            "run_count": int(pd.to_numeric(failure_breakdown_base_df[col], errors="coerce").fillna(0).sum()),
        }
    )

note_flag_summary_df = pd.DataFrame(note_flag_summary_rows).sort_values(
    ["run_count", "note_flag_column"],
    ascending=[False, True],
).reset_index(drop=True)

# per config counts by exact failure reason
if failed_runs_df.empty:
    config_failure_reason_breakdown_df = pd.DataFrame()
else:
    config_failure_reason_breakdown_df = (
        failed_runs_df.assign(
            funnel_failure_reason=failed_runs_df["funnel_failure_reason"].fillna("unknown_failure_reason")
        )
        .pivot_table(
            index="config_label",
            columns="funnel_failure_reason",
            values="run_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    reason_cols = [col for col in config_failure_reason_breakdown_df.columns if col != "config_label"]
    if len(reason_cols) > 0:
        config_failure_reason_breakdown_df["total_failed_runs"] = config_failure_reason_breakdown_df[reason_cols].sum(axis=1)
        config_failure_reason_breakdown_df = config_failure_reason_breakdown_df.sort_values(
            ["total_failed_runs", "config_label"],
            ascending=[False, True],
        ).reset_index(drop=True)

# per config counts by exact funnel step
if failed_runs_df.empty:
    config_failure_step_breakdown_df = pd.DataFrame()
else:
    config_failure_step_breakdown_df = (
        failed_runs_df.assign(
            funnel_failure_step=failed_runs_df["funnel_failure_step"].fillna("unknown_failure_step")
        )
        .pivot_table(
            index="config_label",
            columns="funnel_failure_step",
            values="run_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    step_cols = [col for col in config_failure_step_breakdown_df.columns if col != "config_label"]
    if len(step_cols) > 0:
        config_failure_step_breakdown_df["total_failed_runs"] = config_failure_step_breakdown_df[step_cols].sum(axis=1)
        config_failure_step_breakdown_df = config_failure_step_breakdown_df.sort_values(
            ["total_failed_runs", "config_label"],
            ascending=[False, True],
        ).reset_index(drop=True)

# one compact top line summary
overall_failure_totals_df = pd.DataFrame(
    [
        {
            "total_runs": int(len(failure_breakdown_base_df)),
            "failed_runs": int(len(failed_runs_df)),
            "alive_runs": int(len(alive_runs_df)),
            "failure_rate": float(len(failed_runs_df) / len(failure_breakdown_base_df)) if len(failure_breakdown_base_df) > 0 else np.nan,
        }
    ]
)

# save useful outputs
failure_reason_summary_csv_path = OUTPUT_ROOT / "failure_reason_summary.csv"
failure_step_summary_csv_path = OUTPUT_ROOT / "failure_step_summary.csv"
failure_flag_summary_csv_path = OUTPUT_ROOT / "failure_flag_summary.csv"
config_failure_reason_breakdown_csv_path = OUTPUT_ROOT / "config_failure_reason_breakdown.csv"
config_failure_step_breakdown_csv_path = OUTPUT_ROOT / "config_failure_step_breakdown.csv"

overall_failure_totals_df.to_csv(OUTPUT_ROOT / "overall_failure_totals.csv", index=False)
failure_reason_summary_df.to_csv(failure_reason_summary_csv_path, index=False)
failure_step_summary_df.to_csv(failure_step_summary_csv_path, index=False)
failure_flag_summary_df.to_csv(failure_flag_summary_csv_path, index=False)

if not config_failure_reason_breakdown_df.empty:
    config_failure_reason_breakdown_df.to_csv(config_failure_reason_breakdown_csv_path, index=False)

if not config_failure_step_breakdown_df.empty:
    config_failure_step_breakdown_df.to_csv(config_failure_step_breakdown_csv_path, index=False)

print("overall failure totals")
display(overall_failure_totals_df)

print("failed runs by exact failure reason")
display(failure_reason_summary_df)

print("failed runs by funnel step")
display(failure_step_summary_df)

print("failed runs by boolean failure flag columns")
display(failure_flag_summary_df)

if not note_flag_summary_df.empty:
    print("non failure note flags")
    display(note_flag_summary_df)

if not config_failure_reason_breakdown_df.empty:
    print("per config breakdown by exact failure reason")
    display(config_failure_reason_breakdown_df)

if not config_failure_step_breakdown_df.empty:
    print("per config breakdown by funnel step")
    display(config_failure_step_breakdown_df)

print("csv outputs saved")
print("failure reason summary path", failure_reason_summary_csv_path)
print("failure step summary path", failure_step_summary_csv_path)
print("failure flag summary path", failure_flag_summary_csv_path)
if not config_failure_reason_breakdown_df.empty:
    print("config failure reason breakdown path", config_failure_reason_breakdown_csv_path)
if not config_failure_step_breakdown_df.empty:
    print("config failure step breakdown path", config_failure_step_breakdown_csv_path)

# clean export to google drive with one snapshot folder
#
# this cell creates one timestamped export folder under output root
# and stores raw engineered model ready and summary data separately
#
# large tables are saved as parquet
# summary tables are saved as csv
# one manifest json is also written for traceability

from pathlib import Path
from datetime import datetime
import json
import shutil

if "OUTPUT_ROOT" not in globals():
    raise ValueError("OUTPUT_ROOT is required before exporting")

EXPORTS_ROOT = OUTPUT_ROOT / "exports"
EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)

export_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
export_name = f"export_{export_timestamp}"
export_dir = EXPORTS_ROOT / export_name

raw_dir = export_dir / "raw_merged"
engineered_dir = export_dir / "engineered"
model_ready_dir = export_dir / "model_ready"
summary_dir = export_dir / "summaries"
manifest_dir = export_dir / "manifests"

for folder in [raw_dir, engineered_dir, model_ready_dir, summary_dir, manifest_dir]:
    folder.mkdir(parents=True, exist_ok=True)

export_manifest = {
    "export_name": export_name,
    "export_timestamp": export_timestamp,
    "base_export_dir": str(export_dir),
    "files": [],
    "tables": {},
    "settings": {},
}

def register_table(table_name: str, df: pd.DataFrame):
    if isinstance(df, pd.DataFrame):
        export_manifest["tables"][table_name] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

def save_parquet_if_exists(df_name: str, df_obj, output_path: Path):
    if df_name in globals() and isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
        df_obj.to_parquet(output_path, index=False)
        export_manifest["files"].append(str(output_path))
        register_table(df_name, df_obj)

def save_csv_if_exists(df_name: str, df_obj, output_path: Path):
    if df_name in globals() and isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
        df_obj.to_csv(output_path, index=False)
        export_manifest["files"].append(str(output_path))
        register_table(df_name, df_obj)

# raw merged tables
if "all_runs_tick_df" in globals() and isinstance(all_runs_tick_df, pd.DataFrame) and not all_runs_tick_df.empty:
    raw_path = raw_dir / "all_runs_tick_df.parquet"
    all_runs_tick_df.to_parquet(raw_path, index=False)
    export_manifest["files"].append(str(raw_path))
    register_table("all_runs_tick_df", all_runs_tick_df)

if "all_runs_tick_after_stall_cut_df" in globals() and isinstance(all_runs_tick_after_stall_cut_df, pd.DataFrame) and not all_runs_tick_after_stall_cut_df.empty:
    raw_cut_path = model_ready_dir / "all_runs_tick_after_stall_cut_df.parquet"
    all_runs_tick_after_stall_cut_df.to_parquet(raw_cut_path, index=False)
    export_manifest["files"].append(str(raw_cut_path))
    register_table("all_runs_tick_after_stall_cut_df", all_runs_tick_after_stall_cut_df)

# engineered tables
if "all_runs_features_df" in globals() and isinstance(all_runs_features_df, pd.DataFrame) and not all_runs_features_df.empty:
    feat_path = engineered_dir / "all_runs_features_df.parquet"
    all_runs_features_df.to_parquet(feat_path, index=False)
    export_manifest["files"].append(str(feat_path))
    register_table("all_runs_features_df", all_runs_features_df)

if "all_runs_features_after_stall_cut_df" in globals() and isinstance(all_runs_features_after_stall_cut_df, pd.DataFrame) and not all_runs_features_after_stall_cut_df.empty:
    feat_cut_path = model_ready_dir / "all_runs_features_after_stall_cut_df.parquet"
    all_runs_features_after_stall_cut_df.to_parquet(feat_cut_path, index=False)
    export_manifest["files"].append(str(feat_cut_path))
    register_table("all_runs_features_after_stall_cut_df", all_runs_features_after_stall_cut_df)

# run level and summary tables
summary_objects = {
    "run_funnel_df": "run_funnel_df.csv",
    "stall_run_summary_df": "stall_run_summary_df.csv",
    "failure_reason_summary_df": "failure_reason_summary_df.csv",
    "failure_step_summary_df": "failure_step_summary_df.csv",
    "failure_flag_summary_df": "failure_flag_summary_df.csv",
    "config_failure_reason_breakdown_df": "config_failure_reason_breakdown_df.csv",
    "config_failure_step_breakdown_df": "config_failure_step_breakdown_df.csv",
    "overall_failure_totals_df": "overall_failure_totals_df.csv",
    "build_summary_df": "build_summary_df.csv",
    "build_errors_df": "build_errors_df.csv",
}

for df_name, file_name in summary_objects.items():
    if df_name in globals():
        df_obj = globals()[df_name]
        if isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
            out_path = summary_dir / file_name
            df_obj.to_csv(out_path, index=False)
            export_manifest["files"].append(str(out_path))
            register_table(df_name, df_obj)

# save tracker and manifest copies for reproducibility
if "tracker_df" in globals() and isinstance(tracker_df, pd.DataFrame) and not tracker_df.empty:
    tracker_copy_path = manifest_dir / "tracker_df_snapshot.csv"
    tracker_df.to_csv(tracker_copy_path, index=False)
    export_manifest["files"].append(str(tracker_copy_path))
    register_table("tracker_df", tracker_df)

if "run_manifest" in globals() and isinstance(run_manifest, pd.DataFrame) and not run_manifest.empty:
    run_manifest_copy_path = manifest_dir / "run_manifest_snapshot.csv"
    run_manifest.to_csv(run_manifest_copy_path, index=False)
    export_manifest["files"].append(str(run_manifest_copy_path))
    register_table("run_manifest", run_manifest)

# store useful settings
if "STALL_NOTE_MINUTES" in globals():
    export_manifest["settings"]["STALL_NOTE_MINUTES"] = STALL_NOTE_MINUTES

if "STALL_FAIL_MINUTES" in globals():
    export_manifest["settings"]["STALL_FAIL_MINUTES"] = STALL_FAIL_MINUTES

# save manifest json
manifest_path = manifest_dir / "export_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(export_manifest, f, indent=2)

export_manifest["files"].append(str(manifest_path))

# small convenience pointer to latest export
latest_pointer_path = EXPORTS_ROOT / "latest_export_path.txt"
with open(latest_pointer_path, "w", encoding="utf-8") as f:
    f.write(str(export_dir))

print("export complete")
print("export folder", export_dir)
print("latest export pointer", latest_pointer_path)
print("files written", len(export_manifest["files"]))

display(
    pd.DataFrame(
        {
            "table_name": list(export_manifest["tables"].keys()),
            "rows": [export_manifest["tables"][k]["rows"] for k in export_manifest["tables"].keys()],
            "columns": [export_manifest["tables"][k]["columns"] for k in export_manifest["tables"].keys()],
        }
    ).sort_values("table_name").reset_index(drop=True)
)

# Publish the stable input contract consumed directly by train_v11.py.
# A staging directory is used so a failed parquet write cannot damage the
# previous complete chunk set. Old part files are removed only after every new
# chunk has been written successfully.
MODEL_READY_CHUNKS_DIR = OUTPUT_ROOT / "all_runs_features_model_ready_chunks"
STREAMING_FUNNEL_OUTPUTS_DIR = OUTPUT_ROOT / "streaming_funnel_outputs"
PUBLISH_CHUNK_ROWS = 250_000

if (
    "all_runs_features_after_stall_cut_df" not in globals()
    or not isinstance(all_runs_features_after_stall_cut_df, pd.DataFrame)
    or all_runs_features_after_stall_cut_df.empty
):
    raise ValueError("model-ready feature table is empty; v11 inputs were not published")

if (
    "run_funnel_df" not in globals()
    or not isinstance(run_funnel_df, pd.DataFrame)
    or run_funnel_df.empty
):
    raise ValueError("run_funnel_df is empty; v11 labels were not published")

model_ready_df = all_runs_features_after_stall_cut_df.sort_values(
    ["config_label", "run_id", "canonical_tick"],
    na_position="last",
).reset_index(drop=True)

missing_node_prefixes = [
    subsystem
    for subsystem in NEW_GRAPH_SUBSYSTEMS
    if not any(col.startswith(f"{subsystem}__") for col in model_ready_df.columns)
]
if missing_node_prefixes:
    raise ValueError(
        "v11 model-ready data is missing new graph telemetry for: "
        + ", ".join(missing_node_prefixes)
    )

required_model_columns = {"run_id", "config_label", "canonical_tick"}
missing_model_columns = sorted(required_model_columns - set(model_ready_df.columns))
if missing_model_columns:
    raise ValueError(
        "v11 model-ready data is missing required identifiers: "
        + ", ".join(missing_model_columns)
    )

required_label_columns = {"run_id", "funnel_failed"}
missing_label_columns = sorted(required_label_columns - set(run_funnel_df.columns))
if missing_label_columns:
    raise ValueError(
        "run_funnel_df is missing required label columns: "
        + ", ".join(missing_label_columns)
    )

MODEL_READY_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
STREAMING_FUNNEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

publish_stage_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")
staging_chunks_dir = OUTPUT_ROOT / f".model_ready_chunks_staging_{publish_stage_timestamp}"
staging_chunks_dir.mkdir(parents=True, exist_ok=False)

published_chunk_names = []
for part_number, start_row in enumerate(range(0, len(model_ready_df), PUBLISH_CHUNK_ROWS)):
    stop_row = min(start_row + PUBLISH_CHUNK_ROWS, len(model_ready_df))
    part_name = f"part_{part_number:05d}.parquet"
    part_path = staging_chunks_dir / part_name
    model_ready_df.iloc[start_row:stop_row].to_parquet(part_path, index=False)
    published_chunk_names.append(part_name)

for old_part_path in MODEL_READY_CHUNKS_DIR.glob("part_*.parquet"):
    old_part_path.unlink()

for staged_part_path in sorted(staging_chunks_dir.glob("part_*.parquet")):
    shutil.move(str(staged_part_path), str(MODEL_READY_CHUNKS_DIR / staged_part_path.name))

staging_chunks_dir.rmdir()

funnel_output_path = STREAMING_FUNNEL_OUTPUTS_DIR / "run_funnel_df.csv"
run_funnel_df.sort_values(["config_label", "run_id"], na_position="last").to_csv(
    funnel_output_path,
    index=False,
)

source_config_folders = sorted(
    run_manifest["config_folder_name"].dropna().astype(str).unique().tolist()
)
v11_publish_manifest = {
    "schema_version": "feature_engineering_v2",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "raw_data_root": str(RAW_DATA_ROOT),
    "output_root": str(OUTPUT_ROOT),
    "config_folder_pattern": CONFIG_FOLDER_PATTERN.pattern,
    "source_config_folders": source_config_folders,
    "new_graph_subsystems": list(NEW_GRAPH_SUBSYSTEMS),
    "graph_node_subsystems": list(GRAPH_NODE_SUBSYSTEMS),
    "model_ready_rows": int(len(model_ready_df)),
    "model_ready_columns": int(model_ready_df.shape[1]),
    "run_label_rows": int(len(run_funnel_df)),
    "chunk_rows": PUBLISH_CHUNK_ROWS,
    "chunk_files": published_chunk_names,
    "model_ready_chunks_dir": str(MODEL_READY_CHUNKS_DIR),
    "run_funnel_path": str(funnel_output_path),
}

v11_publish_manifest_path = OUTPUT_ROOT / "feature_engineering_v2_manifest.json"
with open(v11_publish_manifest_path, "w", encoding="utf-8") as f:
    json.dump(v11_publish_manifest, f, indent=2)

print("v11 training inputs published")
print("model-ready chunks", MODEL_READY_CHUNKS_DIR)
print("model-ready rows", len(model_ready_df))
print("model-ready chunk count", len(published_chunk_names))
print("run funnel labels", funnel_output_path)
print("publish manifest", v11_publish_manifest_path)
