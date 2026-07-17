#!/bin/bash -l
#SBATCH --job-name=SF2_Auto
#SBATCH --output=logs/sf_queue_%j.out
#SBATCH --error=logs/sf_queue_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=32g
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

set -Eeuo pipefail

ROOT_DIR="/common/home/rvk22/spaceforge-xai-run2"
SIM_DIR="$ROOT_DIR/Sim"
ACTIVE_JOBS_DIR="$ROOT_DIR/active_jobs"

DATA_ROOT="$ROOT_DIR/data/raw"

TRACKER_CSV="$ROOT_DIR/SpaceForge-xai Job tracker - Analytics.csv"
CONFIG_ARGS_SCRIPT="$ROOT_DIR/scripts/config_row_to_args.py"

MAX_JOBS="${MAX_JOBS:-0}"
SIM_MODE="${SIM_MODE:-wake}"
RUN_SPARTA="${RUN_SPARTA:-OFF}"

cd "$ROOT_DIR"

mkdir -p logs
mkdir -p "$DATA_ROOT"

CONFIG_NAME="${1:-}"

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: missing config name"
    echo "Usage: sbatch Sim/run_orbit3.slurm Config27"
    exit 1
fi

if [ ! -f "$TRACKER_CSV" ]; then
    echo "ERROR: tracker csv not found"
    echo "$TRACKER_CSV"
    exit 1
fi

if [ ! -f "$CONFIG_ARGS_SCRIPT" ]; then
    echo "ERROR: config argument script not found"
    echo "$CONFIG_ARGS_SCRIPT"
    exit 1
fi

if [ ! -d "$ACTIVE_JOBS_DIR" ]; then
    echo "ERROR: active_jobs folder not found"
    echo "$ACTIVE_JOBS_DIR"
    exit 1
fi

if ! PARAM_ARGS_STR=$(python3 "$CONFIG_ARGS_SCRIPT" "$TRACKER_CSV" "$CONFIG_NAME"); then
    echo "ERROR: failed to load parameters for config"
    echo "$CONFIG_NAME"
    exit 1
fi

eval "PARAM_ARGS_ARRAY=($PARAM_ARGS_STR)"

CONFIG_OUT_DIR="$DATA_ROOT/$CONFIG_NAME"
mkdir -p "$CONFIG_OUT_DIR"

echo "$PARAM_ARGS_STR" > "$CONFIG_OUT_DIR/runtime_args_config_level.txt"
cp "$TRACKER_CSV" "$CONFIG_OUT_DIR/analytics_snapshot.csv"

echo "================================================"
echo "SPACEFORGE AUTOMATED CONFIG RUN"
echo "CONFIG NAME: $CONFIG_NAME"
echo "CONFIG OUTPUT: $CONFIG_OUT_DIR"
echo "TRACKER CSV: $TRACKER_CSV"
echo "SIM MODE: $SIM_MODE"
echo "RUN SPARTA: $RUN_SPARTA"
echo "MAX JOBS: $MAX_JOBS"
echo "PARAM ARGS: $PARAM_ARGS_STR"
echo "================================================"

mapfile -t JOB_PATHS < <(find "$ACTIVE_JOBS_DIR" -maxdepth 1 -type f -name "V4_job*.txt" | sort -V)

if [ "${#JOB_PATHS[@]}" -eq 0 ]; then
    echo "ERROR: no V4_job*.txt files found in active_jobs"
    exit 1
fi

EXPECTED_CSVS=(
    "Battery.csv"
    "EffusionCell.csv"
    "HeaterBank.csv"
    "Orbit.csv"
    "PowerBus.csv"
    "ProcessState.csv"
    "ScheduleState.csv"
    "SimulationEngine.csv"
    "SolarArray.csv"
    "substrate.csv"
    "ArrayGimbal.csv"
    "BatteryThermal.csv"
    "CryoPanel.csv"
    "Radiator.csv"
    "SourceInventory.csv"
)

JOB_COUNTER=0

for FILE_PATH in "${JOB_PATHS[@]}"; do

    JOB_COUNTER=$((JOB_COUNTER + 1))

    if [ "$MAX_JOBS" -gt 0 ] && [ "$JOB_COUNTER" -gt "$MAX_JOBS" ]; then
        echo "MAX_JOBS reached. Stopping after $MAX_JOBS job or jobs."
        break
    fi

    JOB_FILE=$(basename "$FILE_PATH")
    CURRENT_ID="${JOB_FILE%.txt}"

    JOB_RUN_ID="${CONFIG_NAME}_${CURRENT_ID}"
    JOB_OUT_DIR="$CONFIG_OUT_DIR/$CURRENT_ID"

    BAD_OLD_LOGGER_DIR="$DATA_ROOT/$JOB_RUN_ID"

    mkdir -p "$JOB_OUT_DIR"

    echo "------------------------------------------------"
    echo "STARTING CONFIG: $CONFIG_NAME"
    echo "STARTING JOB: $CURRENT_ID"
    echo "JOB RUN ID: $JOB_RUN_ID"
    echo "JOB FILE: $JOB_FILE"
    echo "JOB FILE ABSOLUTE PATH: $FILE_PATH"
    echo "JOB OUTPUT: $JOB_OUT_DIR"
    echo "------------------------------------------------"

    for CSV_NAME in "${EXPECTED_CSVS[@]}"; do
        rm -f "$JOB_OUT_DIR/$CSV_NAME"
    done

    rm -f "$JOB_OUT_DIR"/sim_debug_*.log
    rm -f "$JOB_OUT_DIR"/sim_rank_progress_*.log
    rm -f "$JOB_OUT_DIR"/runtime_stdout_stderr.log
    rm -f "$JOB_OUT_DIR"/run_spa.log

    rm -rf "$BAD_OLD_LOGGER_DIR"

    echo "$PARAM_ARGS_STR" > "$JOB_OUT_DIR/runtime_args.txt"
    cp "$TRACKER_CSV" "$JOB_OUT_DIR/analytics_snapshot.csv"
    cp "$FILE_PATH" "$JOB_OUT_DIR/$JOB_FILE"

    python3 - "$JOB_OUT_DIR/params.csv" "${PARAM_ARGS_ARRAY[@]}" <<'PY'
import csv
import sys

out_path = sys.argv[1]
args = sys.argv[2:]

rows = []
i = 0

while i < len(args):
    token = args[i]

    if token.startswith("--"):
        name = token[2:]

        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            value = args[i + 1]
            i += 2
        else:
            value = "true"
            i += 1

        rows.append((name, value))
    else:
        rows.append(("arg", token))
        i += 1

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value"])
    writer.writerows(rows)
PY

    rm -f "$SIM_DIR"/*.csv
    rm -f "$SIM_DIR"/sim_debug_*.log
    rm -f "$SIM_DIR"/sim_rank_progress_*.log
    rm -f "$SIM_DIR"/run_spa.log

    cd "$SIM_DIR"

    set +e

    RUN_ID="$JOB_RUN_ID" \
    OUTPUT_ROOT="$JOB_OUT_DIR" \
    SF_LOG_DIR="$JOB_OUT_DIR" \
    REQUIRE_OUTPUT_ROOT=ON \
    BUILD_DIR="$ROOT_DIR/build_work/${SLURM_JOB_ID}_${JOB_RUN_ID}" \
    MODE="$SIM_MODE" \
    RUN_SPARTA="$RUN_SPARTA" \
    ENABLE_SPARTA=OFF \
    GPU=OFF \
    NP=8 \
    J=8 \
    ./run.sh \
        --nticks 1350 \
        --job-file "$FILE_PATH" \
        "${PARAM_ARGS_ARRAY[@]}" \
        > "$JOB_OUT_DIR/runtime_stdout_stderr.log" 2>&1

    EXIT_CODE=$?

    set -e

    cd "$ROOT_DIR"

    find "$SIM_DIR" -maxdepth 1 -type f -name "*.csv" -exec mv -f {} "$JOB_OUT_DIR/" \;
    find "$SIM_DIR" -maxdepth 1 -type f -name "sim_debug_*.log" -exec mv -f {} "$JOB_OUT_DIR/" \;
    find "$SIM_DIR" -maxdepth 1 -type f -name "sim_rank_progress_*.log" -exec mv -f {} "$JOB_OUT_DIR/" \;
    find "$SIM_DIR" -maxdepth 1 -type f -name "run_spa.log" -exec rm -f {} \;

    rm -f "$JOB_OUT_DIR/run_spa.log"

    if [ -d "$BAD_OLD_LOGGER_DIR" ]; then
        echo "ERROR: logger created a bad old output folder:"
        echo "$BAD_OLD_LOGGER_DIR"
        echo "Your Logger.cpp is still appending RUN_ID instead of using OUTPUT_ROOT directly."
        exit 11
    fi

    MISSING_CSVS=()

    for CSV_NAME in "${EXPECTED_CSVS[@]}"; do
        if [ ! -s "$JOB_OUT_DIR/$CSV_NAME" ]; then
            MISSING_CSVS+=("$CSV_NAME")
        fi
    done

    {
        echo "config_name,$CONFIG_NAME"
        echo "job_id,$CURRENT_ID"
        echo "job_run_id,$JOB_RUN_ID"
        echo "job_file,$JOB_FILE"
        echo "job_file_absolute,$FILE_PATH"
        echo "job_output,$JOB_OUT_DIR"
        echo "sim_mode,$SIM_MODE"
        echo "run_sparta,$RUN_SPARTA"
        echo "exit_code,$EXIT_CODE"
        echo "missing_csv_count,${#MISSING_CSVS[@]}"
        if [ "${#MISSING_CSVS[@]}" -gt 0 ]; then
            printf "missing_csvs,"
            printf "%s " "${MISSING_CSVS[@]}"
            printf "\n"
        else
            echo "missing_csvs,"
        fi
    } > "$JOB_OUT_DIR/job_status.txt"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "FAILED: $CONFIG_NAME $CURRENT_ID with exit code $EXIT_CODE"
        echo "See:"
        echo "$JOB_OUT_DIR/runtime_stdout_stderr.log"
        exit "$EXIT_CODE"
    fi

    if [ "${#MISSING_CSVS[@]}" -gt 0 ]; then
        echo "FAILED: $CONFIG_NAME $CURRENT_ID completed but did not produce all subsystem CSVs."
        echo "Missing CSVs:"
        printf "  %s\n" "${MISSING_CSVS[@]}"
        echo "See:"
        echo "$JOB_OUT_DIR/runtime_stdout_stderr.log"
        exit 10
    fi

    echo "DONE: $CONFIG_NAME $CURRENT_ID"

done

echo "================================================"
echo "ALL REQUESTED JOBS COMPLETE FOR CONFIG: $CONFIG_NAME"
echo "OUTPUT FOLDER: $CONFIG_OUT_DIR"
echo "================================================"