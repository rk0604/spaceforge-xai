#!/usr/bin/env bash
set -euo pipefail

# This file lives in Sim, so the repository root is one level up.
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SIM_DIR}/.." && pwd)"

# Each run gets an identifier used for logs and reproducibility.
: "${RUN_ID:=run_default}"

RUN_ID_SAFE="$(printf '%s' "${RUN_ID}" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
if [ -z "${RUN_ID_SAFE}" ]; then
  RUN_ID_SAFE="run_default"
fi

# Slurm must pass OUTPUT_ROOT explicitly.
# This prevents accidental loose folders in data/raw.
: "${REQUIRE_OUTPUT_ROOT:=OFF}"

if [ -z "${OUTPUT_ROOT:-}" ]; then
  if [ "${REQUIRE_OUTPUT_ROOT}" = "ON" ]; then
    echo "[run.sh] ERROR: OUTPUT_ROOT is required but was not provided."
    echo "[run.sh] This prevents accidental writes to data/raw/${RUN_ID_SAFE}."
    exit 2
  fi

  OUTPUT_ROOT="${REPO_ROOT}/data/raw/${RUN_ID_SAFE}"
fi

mkdir -p "${OUTPUT_ROOT}"

# Build directory.
# Slurm should usually pass this explicitly.
if [ -z "${BUILD_DIR:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    BUILD_DIR="${REPO_ROOT}/build_work/${SLURM_JOB_ID}_${RUN_ID_SAFE}"
  else
    BUILD_DIR="${SIM_DIR}/build"
  fi
fi

# GPU selector.
# GPU=ON uses the GPU SPARTA path.
# GPU=OFF uses the CPU SPARTA path.
: "${GPU:=OFF}"

GPU_UPPER="$(printf '%s' "${GPU}" | tr '[:lower:]' '[:upper:]')"

if [ "${GPU_UPPER}" = "ON" ]; then
  SPARTA_DIR_DEFAULT="$HOME/opt/sparta/build-gpu/src"
  SPARTA_EXE_DEFAULT="$HOME/opt/sparta/build-gpu/src/spa_"
  BASE_SPARTA_ARGS="-k on g 1 -sf kk"
  GPU_MODE_LABEL="gpu"
else
  SPARTA_DIR_DEFAULT="$HOME/opt/sparta/src"
  SPARTA_EXE_DEFAULT="$HOME/opt/sparta/src/spa_"
  BASE_SPARTA_ARGS=""
  GPU_MODE_LABEL="cpu"
fi

# Libsparta and compiler paths.
SPARTA_DIR="${SPARTA_DIR:-$SPARTA_DIR_DEFAULT}"

: "${CC:=mpicc}"
: "${CXX:=mpicxx}"

# ENABLE_SPARTA controls build linkage.
# ENABLE_SPARTA=OFF means the simulator uses the external executable shim.
# RUN_SPARTA controls whether the external executable is the real SPARTA binary.
# RUN_SPARTA=OFF keeps wake scheduler mode alive but replaces SPARTA execution with /bin/true.
: "${ENABLE_SPARTA:=OFF}"
: "${RUN_SPARTA:=OFF}"
: "${CMAKE_VERBOSE_MAKEFILE:=ON}"

RUN_SPARTA_UPPER="$(printf '%s' "${RUN_SPARTA}" | tr '[:lower:]' '[:upper:]')"

# Runtime execution settings.
: "${NP:=1}"
: "${MODE:=wake}"
: "${WAKE_DECK:=in.wake}"

# Orbit and environment knobs passed into the wake deck when real SPARTA is enabled.
: "${PTORR_TARGET:=1.0e-7}"
: "${PCUP_TORR:=9.0e-9}"
: "${FWAfer_CM2S:=1.0e14}"

: "${CUP_BASE_SCALE:=1.0}"
: "${CUP_AMP_SCALE:=0.50}"
: "${CUP_PHASE0:=0.0}"

# Make input absolute so the simulator can run from any working directory.
INPUT_SUBDIR="${INPUT_SUBDIR:-${REPO_ROOT}/input}"

# Make the wake deck absolute so the simulator can run from OUTPUT_ROOT.
if [[ "${WAKE_DECK}" = /* ]]; then
  WAKE_DECK_PATH="${WAKE_DECK}"
else
  WAKE_DECK_PATH="${SIM_DIR}/${WAKE_DECK}"
fi

# Configure the external SPARTA executable.
# In C++ only dataset mode, WakeChamber still follows scheduler code paths,
# but the external SPARTA call is replaced by /bin/true.
if [ "${RUN_SPARTA_UPPER}" = "OFF" ]; then
  SPARTA_EXE="/bin/true"
  SPARTA_EXTRA_ARGS=""
else
  SPARTA_EXE="${SPARTA_EXE:-$SPARTA_EXE_DEFAULT}"

  ORBIT_SPARTA_ARGS=(
    -var pTorrTarget    "${PTORR_TARGET}"
    -var Pcup_Torr      "${PCUP_TORR}"
    -var cup_base_scale "${CUP_BASE_SCALE}"
    -var cup_amp_scale  "${CUP_AMP_SCALE}"
    -var phase0         "${CUP_PHASE0}"
    -var runID          "${RUN_ID}"
  )

  if [ -z "${SPARTA_EXTRA_ARGS:-}" ]; then
    SPARTA_EXTRA_ARGS="${BASE_SPARTA_ARGS} ${ORBIT_SPARTA_ARGS[*]}"
  fi
fi

export SPARTA_EXE
export SPARTA_EXTRA_ARGS
export OUTPUT_ROOT
export SF_LOG_DIR="${OUTPUT_ROOT}"

echo "[run.sh] REPO_ROOT         = ${REPO_ROOT}"
echo "[run.sh] SIM_DIR           = ${SIM_DIR}"
echo "[run.sh] BUILD_DIR         = ${BUILD_DIR}"
echo "[run.sh] GPU               = ${GPU_UPPER} (${GPU_MODE_LABEL})"
echo "[run.sh] SPARTA_DIR        = ${SPARTA_DIR}"
echo "[run.sh] ENABLE_SPARTA     = ${ENABLE_SPARTA}"
echo "[run.sh] RUN_SPARTA        = ${RUN_SPARTA_UPPER}"
echo "[run.sh] SPARTA_EXE        = ${SPARTA_EXE}"
echo "[run.sh] SPARTA_EXTRA_ARGS = ${SPARTA_EXTRA_ARGS}"
echo "[run.sh] MODE              = ${MODE}"
echo "[run.sh] WAKE_DECK         = ${WAKE_DECK}"
echo "[run.sh] WAKE_DECK_PATH    = ${WAKE_DECK_PATH}"
echo "[run.sh] INPUT_SUBDIR      = ${INPUT_SUBDIR}"
echo "[run.sh] NP                = ${NP}"
echo "[run.sh] RUN_ID            = ${RUN_ID}"
echo "[run.sh] OUTPUT_ROOT       = ${OUTPUT_ROOT}"
echo "[run.sh] PTORR_TARGET      = ${PTORR_TARGET}"
echo "[run.sh] PCUP_TORR         = ${PCUP_TORR}"
echo "[run.sh] FWAfer_CM2S       = ${FWAfer_CM2S}"
echo "[run.sh] CUP_BASE_SCALE    = ${CUP_BASE_SCALE}"
echo "[run.sh] CUP_AMP_SCALE     = ${CUP_AMP_SCALE}"
echo "[run.sh] CUP_PHASE0        = ${CUP_PHASE0}"
echo "[run.sh] EXTRA SIM ARGS    = $*"
echo

# Configure and build.
# The build folder is removed first so source edits are always reflected.
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

CC="$CC" CXX="$CXX" cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DSPARTA_DIR="${SPARTA_DIR}" \
  -DENABLE_SPARTA="${ENABLE_SPARTA}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_VERBOSE_MAKEFILE="${CMAKE_VERBOSE_MAKEFILE}"

cmake --build "${BUILD_DIR}" -j "${J:-8}"

SIM_EXE="${BUILD_DIR}/Sim/sim"

if [ ! -x "${SIM_EXE}" ]; then
  echo "[run.sh] ERROR: simulator binary was not built or is not executable."
  echo "[run.sh] Missing binary: ${SIM_EXE}"
  exit 3
fi

# Run from OUTPUT_ROOT.
# Logger.cpp should also use OUTPUT_ROOT or SF_LOG_DIR directly.
cd "${OUTPUT_ROOT}"

echo "[run.sh] Runtime working directory:"
pwd
echo

unset DISPLAY XAUTHORITY

env -u DISPLAY -u XAUTHORITY mpirun -np "${NP}" "${SIM_EXE}" \
  --mode "${MODE}" \
  --wake-deck "${WAKE_DECK_PATH}" \
  --input-subdir "${INPUT_SUBDIR}" \
  "$@"

echo
echo "[run.sh] Finished simulator run."
echo "[run.sh] CSV files in OUTPUT_ROOT after run:"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name "*.csv" -printf "  %f\n" | sort