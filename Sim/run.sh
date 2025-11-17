#!/usr/bin/env bash
set -euo pipefail

# This file lives in Sim/, so repo root is one level up.
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SIM_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${SIM_DIR}/build}"

# -------------------------
# GPU / CPU selector
#   GPU=ON  -> use build-gpu (Kokkos)
#   GPU=OFF -> use CPU build
# -------------------------
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
  BASE_SPARTA_ARGS=""        # no Kokkos flags in CPU mode
  GPU_MODE_LABEL="cpu"
fi

# Libsparta + include (can be overridden explicitly via env)
SPARTA_DIR="${SPARTA_DIR:-$SPARTA_DIR_DEFAULT}"

: "${CC:=mpicc}"
: "${CXX:=mpicxx}"

# ENABLE_SPARTA=OFF  -> external-binary shim only
# ENABLE_SPARTA=ON   -> link against libsparta (C API)
: "${ENABLE_SPARTA:=OFF}"
: "${CMAKE_VERBOSE_MAKEFILE:=ON}"

: "${NP:=1}"
: "${MODE:=wake}"
: "${WAKE_DECK:=in.wake}"

# -------------------------
# Per-run identifier for output subdirs under data/raw/
# -------------------------
: "${RUN_ID:=run_default}"

# -------------------------
# Orbit / environment knobs (passed into in.wake via -var ...)
# -------------------------
: "${PTORR_TARGET:=1.0e-7}"     # freestream ambient p_inf [Torr]
: "${PCUP_TORR:=9.0e-9}"        # cupola target wake add [Torr]
: "${FWAfer_CM2S:=1.0e14}"      # wafer-center MBE flux [cm^-2 s^-1]

: "${CUP_BASE_SCALE:=1.0}"      # baseline outgassing scale
: "${CUP_AMP_SCALE:=0.50}"      # sinusoidal amplitude
: "${CUP_PHASE0:=0.0}"          # phase offset for sinusoid

# Make input absolute so shim / library can cd into it safely
INPUT_SUBDIR="${INPUT_SUBDIR:-${REPO_ROOT}/input}"

# Per-run output root (both C++ CSVs and SPARTA CSVs can live under here)
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/raw/${RUN_ID}}"
mkdir -p "${OUTPUT_ROOT}"

# External SPARTA binary (used only if ENABLE_SPARTA=OFF)
SPARTA_EXE="${SPARTA_EXE:-$SPARTA_EXE_DEFAULT}"

# Build the -var overrides that in.wake expects
ORBIT_SPARTA_ARGS=(
  -var pTorrTarget    "${PTORR_TARGET}"
  -var Pcup_Torr      "${PCUP_TORR}"
  -var Fwafer_cm2s    "${FWAfer_CM2S}"
  -var cup_base_scale "${CUP_BASE_SCALE}"
  -var cup_amp_scale  "${CUP_AMP_SCALE}"
  -var phase0         "${CUP_PHASE0}"
  -var runID          "${RUN_ID}"
)

# If caller didn’t override SPARTA_EXTRA_ARGS, build it
if [ -z "${SPARTA_EXTRA_ARGS:-}" ]; then
  SPARTA_EXTRA_ARGS="${BASE_SPARTA_ARGS} ${ORBIT_SPARTA_ARGS[*]}"
fi

export SPARTA_EXE SPARTA_EXTRA_ARGS

echo "[run.sh] REPO_ROOT         = ${REPO_ROOT}"
echo "[run.sh] BUILD_DIR         = ${BUILD_DIR}"
echo "[run.sh] GPU               = ${GPU_UPPER} (${GPU_MODE_LABEL})"
echo "[run.sh] SPARTA_DIR        = ${SPARTA_DIR}"
echo "[run.sh] ENABLE_SPARTA     = ${ENABLE_SPARTA}"
echo "[run.sh] SPARTA_EXE        = ${SPARTA_EXE}"
echo "[run.sh] SPARTA_EXTRA_ARGS = ${SPARTA_EXTRA_ARGS}"
echo "[run.sh] MODE              = ${MODE}"
echo "[run.sh] WAKE_DECK         = ${WAKE_DECK}"
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
echo

# --- Configure + Build ---
rm -rf "${BUILD_DIR}"
CC="$CC" CXX="$CXX" cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DSPARTA_DIR="${SPARTA_DIR}" \
  -DENABLE_SPARTA="${ENABLE_SPARTA}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_VERBOSE_MAKEFILE="${CMAKE_VERBOSE_MAKEFILE}"

cmake --build "${BUILD_DIR}" -j "${J:-8}"

# --- Run ---
cd "${BUILD_DIR}"
unset DISPLAY XAUTHORITY
env -u DISPLAY -u XAUTHORITY mpirun -np "${NP}" ./Sim/sim \
  --mode "${MODE}" \
  --wake-deck "${WAKE_DECK}" \
  --input-subdir "${INPUT_SUBDIR}" \
  "$@"
