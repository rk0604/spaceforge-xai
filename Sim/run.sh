#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ---- Configure & build (no 'module' calls) ----
SPARTA_DIR="${SPARTA_DIR:-$HOME/opt/sparta}"   # adjust if different
: "${CC:=mpicc}"; : "${CXX:=mpicxx}"

CC="$CC" CXX="$CXX" cmake -DSPARTA_DIR="$SPARTA_DIR" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j

# ---- Run ----
unset DISPLAY XAUTHORITY
SPARTA_EXE="${SPARTA_EXE:-$SPARTA_DIR/sparta}"

mpirun -np "${NP:-4}" ./sim_app \
  --mode dual --split "${SPLIT:-2}" \
  --wake-deck ../input/in.wake \
  --eff-deck  ../input/in.effusion \
  --input-subdir ../input \
  --couple-every 10 \
  --sparta-block 200 \
  ${SPARTA_EXE:+--sparta-exe "$SPARTA_EXE"}
