# SpaceForge-XAI: SPARTA + C++ Harness Quick Reference

This README explains the common commands you’ve been using to:
- build and run **SPARTA alone** (GPU build), and  
- run the **C++ harness + SPARTA** via `run.sh`, including how CLI flags and environment variables are parsed.

---

## 1. Running SPARTA Alone (GPU build)

These commands are for rebuilding SPARTA with Kokkos/CUDA and running the `in.wake` deck directly, **without** the C++ harness.

### 1.1. Reset and clean the SPARTA source tree

```bash
cd ~/opt/sparta/src

# Clear all Makefile package selections (what CMake is asking for)
make no-all

# (Recommended) also clean old object files & build dirs
make clean-all   # this may take a little while
```

**What this does:**  
- `cd ~/opt/sparta/src` goes to the SPARTA source directory.  
- `make no-all` clears any previously selected packages in the old Makefile-style build (this keeps CMake from picking up stale Makefile settings).  
- `make clean-all` removes old object files and previous builds, giving you a clean slate before switching to the CMake-based GPU build.

> You’re basically “resetting” the legacy build so it doesn’t interfere with the CMake+Kokkos configuration.

---

### 1.2. Set up CUDA + SPARTA environment variables

```bash
cd ~/opt

# --- CUDA toolchain env ---
export NVCC=/usr/local/cuda-12.6/bin/nvcc
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)

# --- SPARTA paths ---
export SPARTA_ROOT="$HOME/opt/sparta"
export SPARTA_BUILD="$SPARTA_ROOT/build-gpu"
export NVCC_WRAPPER="$SPARTA_ROOT/lib/kokkos/bin/nvcc_wrapper"
```

**What this does:**  
- Points **NVCC** to the CUDA 12.6 compiler.  
- Extends `PATH` so `nvcc` is found on the command line.  
- Sets `CUDA_HOME` to your CUDA installation root.  
- `NVCC_WRAPPER_DEFAULT_COMPILER` tells Kokkos’ `nvcc_wrapper` which host C++ compiler to use (here, the `g++` from your environment).  
- `SPARTA_ROOT` and `SPARTA_BUILD` define where SPARTA lives and where the **GPU build directory** will be created.  
- `NVCC_WRAPPER` points to Kokkos’ compiler wrapper script, which knows how to compile CUDA+C++ with the right flags.

---

### 1.3. Sanity checks and fresh build directory

```bash
# sanity (should still exist)
ls -l "$SPARTA_ROOT/cmake/CMakeLists.txt"
ls -l "$NVCC_WRAPPER"

# fresh build dir
rm -rf "$SPARTA_BUILD"
```

**What this does:**  
- `ls` checks that the CMake config file and Kokkos `nvcc_wrapper` actually exist, catching path typos early.  
- `rm -rf "$SPARTA_BUILD"` deletes any previous GPU build to guarantee a clean CMake configure step.

---

### 1.4. Configure SPARTA for GPU with CMake

```bash
cmake -S "$SPARTA_ROOT/cmake" -B "$SPARTA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_KOKKOS=ON \
  -DPKG_KOKKOS=ON \
  -DPKG_FFT=ON \
  -DFFT=KISS \
  -DFFT_KOKKOS=KISS \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF
```

**What this does:**  
- `-S "$SPARTA_ROOT/cmake"` tells CMake where the SPARTA CMakeLists live (the “source” dir).  
- `-B "$SPARTA_BUILD"` sets the build directory (out-of-source build).  
- `-DCMAKE_CUDA_COMPILER="$NVCC"` uses your CUDA compiler.  
- `-DCMAKE_CXX_COMPILER="$NVCC_WRAPPER"` uses Kokkos’ `nvcc_wrapper` for all C++ compilation.  
- `-DCMAKE_CUDA_ARCHITECTURES=86` targets your GPU (e.g., RTX 40xx class).  
- `BUILD_KOKKOS`, `PKG_KOKKOS`, `Kokkos_ENABLE_*` enable the Kokkos/CUDA back end.  
- `PKG_FFT`, `FFT=KISS`, `FFT_KOKKOS=KISS` enable a FFT package backed by the KISS FFT implementation.

> This step wires SPARTA to build in **GPU mode** with Kokkos enabled.

---

### 1.5. Build SPARTA (GPU)

```bash
# ---- BUILD ----
cmake --build "$SPARTA_BUILD" -j 8
```

**What this does:**  
- Invokes the actual compilation for the GPU build directory with up to 8 parallel compile jobs (`-j 8`).  
- Produces the `spa_` binary under `"$SPARTA_BUILD/src"`.

---

### 1.6. Run the GPU SPARTA wake deck directly

```bash
cd ~/spaceforge-xai/input 

env -u DISPLAY -u XAUTHORITY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
mpirun -np 1 "$HOME/opt/sparta/build-gpu/src/spa_" \
  -in in.wake -k on g 1 -sf kk \
| tee ~/spaceforge-xai/Sim/build/run_spa_gpu.log
```

**What this does:**  
- `cd ~/spaceforge-xai/input` goes to the folder with your SPARTA deck `in.wake`.  
- `env -u DISPLAY -u XAUTHORITY` strips GUI-related vars so SPARTA runs cleanly headless.  
- `CUDA_VISIBLE_DEVICES=0` pins SPARTA to GPU 0.  
- `OMP_NUM_THREADS=1` forces 1 OpenMP thread per MPI rank (useful for consistency).  
- `mpirun -np 1` runs SPARTA on 1 MPI rank.  
- `"$HOME/opt/sparta/build-gpu/src/spa_"` is the GPU SPARTA executable.  
- `-in in.wake` tells SPARTA which input deck to run.  
- `-k on g 1 -sf kk` enables Kokkos (`-k on`), uses 1 GPU per rank (`g 1`), and selects Kokkos style fixes (`-sf kk`).  
- `| tee ...run_spa_gpu.log` records the full SPARTA output to `run_spa_gpu.log` while still echoing it to the terminal.

> This is your “SPARTA only” GPU benchmark command for the wake deck.

---

### 1.7. Timing notes: `fnum` and CPU vs GPU speed

```text
fnumm: 13
cpu = 26s per 1k steps CPU
GPU = 34s per 1k steps GPU 

fnum: 12
cpu = 182s per 1k steps
gpu = 240s per 1k steps
 - GPu(new in.wake optimized for gpu) = 150s per 1k steps
```

**What this means:**  
- `fnum` / `fnumm` correspond to different SPARTA parameter choices (e.g., macro-particles per cell or related settings).  
- You measured wall-clock time per 1000 SPARTA steps for **CPU vs GPU** configurations.  
- These notes help you compare different deck tunings (e.g., an optimized GPU version of `in.wake` that reduces time from 240 s → 150 s per 1k steps).

---

## 2. Running the C++ Harness + SPARTA (via `run.sh`)

This section covers how to run the **C++ simulation harness** (`Sim/sim`) with or without SPARTA, and how the arguments are wired through `run.sh`.

### 2.1. Wake deck via harness (C++ + SPARTA together)

```bash
# ----------------- in.wake variable input commands
cd ~/spaceforge-xai/Sim
RUN_ID=test_low_alt PTORR_TARGET=3.0e-7 ENABLE_SPARTA=ON MODE=wake GPU=OFF ./run.sh
# or just omit GPU entirely; default is OFF
```

**What this does:**  
- `cd ~/spaceforge-xai/Sim` moves into the Sim/ directory where `run.sh` lives.  
- `RUN_ID=test_low_alt` tags this run with the ID `test_low_alt`. All raw outputs go under `data/raw/test_low_alt`.  
- `PTORR_TARGET=3.0e-7` overrides the default freestream ambient pressure (Torr) passed into `in.wake`.  
- `ENABLE_SPARTA=ON` tells CMake and the harness to link against **libsparta** and use the C API (instead of external-binary mode).  
- `MODE=wake` chooses the simulation mode the C++ executable will use (`--mode wake` internally).  
- `GPU=OFF` tells `run.sh` to use the **CPU SPARTA build** (`$HOME/opt/sparta/src/spa_`) if it needs an external binary.  
- `./run.sh` runs the script with all of the above environment variables applied.

> This is your **“wake mode, C++ + SPARTA”** run, with a custom pressure and a named run ID for bookkeeping.

To use the GPU SPARTA build from the harness, simply set `GPU=ON`:

```bash
RUN_ID=test_low_alt_gpu PTORR_TARGET=3.0e-7 ENABLE_SPARTA=OFF MODE=wake GPU=ON ./run.sh
```

Here, `GPU=ON` switches SPARTA paths to `build-gpu`, and `ENABLE_SPARTA=OFF` means the harness treats SPARTA as an **external executable** (using `SPARTA_EXE` + `SPARTA_EXTRA_ARGS`).

---

### 2.2. Power-mode: C++ harness only (no SPARTA)

```bash
## run power mode to isolate c++ sim and run it solo without sparta
# Rebuild once (from repo root, as usual)
cd ~/spaceforge-xai
rm -rf build && mkdir build && cd build
cmake -DSPARTA_DIR="$HOME/opt/sparta/src" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
cd ~/spaceforge-xai/Sim

RUN_ID=debug_power_only \
ENABLE_SPARTA=OFF \
MODE=power \
./run.sh
```

**What this does:**  
1. **Manual rebuild** (equivalent to what `run.sh` does automatically):  
   - `rm -rf build && mkdir build && cd build` clears and recreates the top-level build directory.  
   - `cmake -DSPARTA_DIR=...` configures the project in Release mode.  
   - `cmake --build . -j` compiles the C++ harness.  

2. **Run the harness in power-only mode:**  
   - `RUN_ID=debug_power_only` keeps the outputs in `data/raw/debug_power_only`.  
   - `ENABLE_SPARTA=OFF` disables libsparta linkage and external SPARTA calls.  
   - `MODE=power` tells the harness to run its internal **“power” mode**, which isolates the C++ logic (e.g., orbital/MBE power model) without SPARTA.  
   - `./run.sh` then configures/builds (again, but that’s okay) and runs the `Sim/sim` binary with `--mode power`.

> Use this when you want to debug the harness itself or prototype the power model without waiting for a DSMC wake solution.

---

## 3. Understanding `run.sh`

`run.sh` is the central script that:

1. Chooses **GPU vs CPU** SPARTA binaries.  
2. Picks **external-binary vs libsparta** mode.  
3. Configures CMake with the right SPARTA paths and options.  
4. Builds the C++ harness and then launches `Sim/sim` with the correct CLI arguments.

Below is a breakdown of the important sections and how to control them from the command line.

### 3.1. Basic layout

```bash
#!/usr/bin/env bash
set -euo pipefail

# This file lives in Sim/, so repo root is one level up.
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SIM_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${SIM_DIR}/build}"
```

- `set -euo pipefail` makes the script exit on errors and undefined variables.  
- `SIM_DIR` is the directory containing `run.sh`.  
- `REPO_ROOT` is the repo top-level (`..` from `Sim/`).  
- `BUILD_DIR` defaults to `Sim/build` unless you override it:  
  ```bash
  BUILD_DIR=/some/other/dir ./run.sh
  ```

---

### 3.2. GPU / CPU selector

```bash
# GPU / CPU selector
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
```

- If you **don’t** specify `GPU`, it defaults to `OFF` (CPU mode).  
- `GPU=ON ./run.sh` switches to the **GPU SPARTA build** under `build-gpu/src` and adds `-k on g 1 -sf kk` to SPARTA arguments by default.  
- `SPARTA_DIR_DEFAULT` and `SPARTA_EXE_DEFAULT` point to the correct SPARTA binaries for the chosen mode.  
- `BASE_SPARTA_ARGS` is a baseline set of SPARTA flags used only when running SPARTA as an external binary.

You can override `SPARTA_DIR` or `SPARTA_EXE` explicitly if needed:

```bash
SPARTA_EXE=/custom/path/spa_ GPU=ON ./run.sh
```

---

### 3.3. Compiler and SPARTA linkage mode

```bash
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
```

- `CC` and `CXX` default to MPI compilers (`mpicc`, `mpicxx`) but can be overridden.  
- `ENABLE_SPARTA` chooses **how** the harness talks to SPARTA:  
  - `ENABLE_SPARTA=ON`: use **libsparta C API** (linked in).  
  - `ENABLE_SPARTA=OFF`: harness expects an **external SPARTA binary** (using `SPARTA_EXE` + env args).  
- `NP` is the number of MPI ranks for the harness run (`mpirun -np "${NP}"`).  
- `MODE` is the logical sim mode: `wake`, `power`, `legacy`, etc.  
- `WAKE_DECK` is the filename of the SPARTA deck (relative to `INPUT_SUBDIR`).

Examples:

```bash
# 4 MPI ranks, GPU SPARTA, wake mode
NP=4 GPU=ON MODE=wake ENABLE_SPARTA=OFF ./run.sh

# legacy single-instance mode with libsparta
ENABLE_SPARTA=ON MODE=legacy ./run.sh
```

---

### 3.4. Run IDs and orbit/environment knobs

```bash
: "${RUN_ID:=run_default}"

: "${PTORR_TARGET:=1.0e-7}"     # freestream ambient p_inf [Torr]
: "${PCUP_TORR:=9.0e-9}"        # cupola target wake add [Torr]
: "${FWAfer_CM2S:=1.0e14}"      # wafer-center MBE flux [cm^-2 s^-1]

: "${CUP_BASE_SCALE:=1.0}"      # baseline outgassing scale
: "${CUP_AMP_SCALE:=0.50}"      # sinusoidal amplitude
: "${CUP_PHASE0:=0.0}"          # phase offset for sinusoid
```

- `RUN_ID` labels this run and is used to create `data/raw/${RUN_ID}` as an output root.  
- The other environment variables capture orbital and chamber parameters that you can vary from the CLI.  
  - `PTORR_TARGET`: target free-stream pressure.  
  - `PCUP_TORR`: additional wake pressure at the cupola.  
  - `FWAfer_CM2S`: wafer-center flux.  
  - `CUP_BASE_SCALE`, `CUP_AMP_SCALE`, `CUP_PHASE0`: control a sinusoidal outgassing model.

Example of changing these on the fly:

```bash
RUN_ID=orbit_sweep \
PTORR_TARGET=5.0e-8 \
PCUP_TORR=5.0e-9 \
CUP_AMP_SCALE=0.8 \
./run.sh
```

---

### 3.5. Input/output directories

```bash
# Make input absolute so shim / library can cd into it safely
INPUT_SUBDIR="${INPUT_SUBDIR:-${REPO_ROOT}/input}"

# Per-run output root (both C++ CSVs and SPARTA CSVs can live under here)
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/raw/${RUN_ID}}"
mkdir -p "${OUTPUT_ROOT}"
```

- `INPUT_SUBDIR` tells the harness where to look for decks and related inputs. Default is `<repo>/input`. Override like this:  
  ```bash
  INPUT_SUBDIR=/some/other/inputs ./run.sh
  ```
- `OUTPUT_ROOT` is where both the C++ harness and SPARTA outputs can be written. Default is `data/raw/<RUN_ID>`. The script ensures the directory exists.

---

### 3.6. SPARTA executable + deck variables

```bash
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
```

- `SPARTA_EXE` can be overridden but defaults to the GPU or CPU path chosen earlier.  
- `ORBIT_SPARTA_ARGS` is an array of `-var` assignments passed into `in.wake`. This is how CLI env knobs become SPARTA variables.  
- If you don’t set `SPARTA_EXTRA_ARGS` yourself, `run.sh` builds it from `BASE_SPARTA_ARGS` and `ORBIT_SPARTA_ARGS`.  
- Both `SPARTA_EXE` and `SPARTA_EXTRA_ARGS` are exported so that the C++ harness or shim can launch SPARTA with consistent settings.

Override example:

```bash
SPARTA_EXTRA_ARGS="-echo screen -var runID custom123" ./run.sh
```

In this case, the script won’t rebuild `SPARTA_EXTRA_ARGS`; it uses what you provide.

---

### 3.7. Diagnostics

```bash
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
```

- Prints out all key settings for this run.  
- Helpful for debugging environment overrides and ensuring your CLI flags are doing what you think.

---

### 3.8. Configure + build the C++ harness

```bash
# --- Configure + Build ---
rm -rf "${BUILD_DIR}"
CC="$CC" CXX="$CXX" cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DSPARTA_DIR="${SPARTA_DIR}" \
  -DENABLE_SPARTA="${ENABLE_SPARTA}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_VERBOSE_MAKEFILE="${CMAKE_VERBOSE_MAKEFILE}"

cmake --build "${BUILD_DIR}" -j "${J:-8}"
```

- `rm -rf "${BUILD_DIR}"` ensures a clean build each time you run the script (no stale artifacts).  
- CMake is invoked with:  
  - `SPARTA_DIR` so the project can find SPARTA headers and libraries.  
  - `ENABLE_SPARTA` as a CMake option to toggle libsparta integration.  
  - `CMAKE_BUILD_TYPE=Release` for optimized builds.  
  - `CMAKE_VERBOSE_MAKEFILE` in case you want detailed compile logs.  
- `cmake --build ... -j "${J:-8}"` compiles using `J` parallel jobs (defaults to 8). Override with:  
  ```bash
  J=4 ./run.sh
  ```

---

### 3.9. Running the harness (CLI args and parsing)

```bash
# --- Run ---
cd "${BUILD_DIR}"
unset DISPLAY XAUTHORITY
env -u DISPLAY -u XAUTHORITY mpirun -np "${NP}" ./Sim/sim \
  --mode "${MODE}" \
  --wake-deck "${WAKE_DECK}" \
  --input-subdir "${INPUT_SUBDIR}" \
  "$@"
```

- `cd "${BUILD_DIR}"` goes into the CMake build directory where `Sim/sim` was produced.  
- GUI-related variables are unset for headless runs.  
- `mpirun -np "${NP}"` launches the harness with the requested number of MPI ranks.  
- `./Sim/sim` is the C++ executable.  
- `--mode "${MODE}"`, `--wake-deck "${WAKE_DECK}"`, `--input-subdir "${INPUT_SUBDIR}"` are **CLI arguments parsed by your C++ `main.cpp` / argument parser**.  
- `"$@"` forwards any additional command-line arguments you provide after `run.sh` to the C++ program.

**Examples:**

```bash
# 1) Default wake mode, extra C++ flags
./run.sh --nticks 500 --sparta-block 200

# 2) Wake mode with explicit deck and GPU SPARTA via external binary
GPU=ON ENABLE_SPARTA=OFF MODE=wake WAKE_DECK=in.wake_harness ./run.sh --nticks 1000

# 3) Power-only mode with 4 MPI ranks
NP=4 MODE=power ENABLE_SPARTA=OFF ./run.sh --nticks 2000
```

Inside `Sim/sim`, your argument parser sees:

- `--mode` (e.g., `"wake"` or `"power"`), which selects the control flow.  
- `--wake-deck` (the file name it passes to SPARTA if enabled).  
- `--input-subdir` (root path for decks).  
- Any extra options like `--nticks`, `--sparta-block`, etc., which you use to set timestep counts, coupling intervals, etc.

---

## 4. Mental Model Summary

- **Environment variables** (e.g., `GPU`, `RUN_ID`, `PTORR_TARGET`, `MODE`) are read by `run.sh` to configure both CMake and SPARTA deck parameters.  
- **`run.sh`** always:  
  1. Chooses SPARTA build and flags (GPU vs CPU).  
  2. Wires orbit/chamber parameters into `in.wake` via `-var` overrides.  
  3. Configures and builds the C++ harness.  
  4. Launches `Sim/sim` with CLI flags that reflect your chosen mode and deck.  
- **CLI args after `run.sh`** (e.g., `--nticks`) go directly into the C++ executable via `$@`, letting you tune timestep counts and coupling patterns without editing the script.

Use this README as your “run cookbook” when you forget which knob lives where or how to switch between:

- **SPARTA-only GPU benchmarks**,  
- **C++ + SPARTA coupled wake runs**, and  
- **C++ power-only experiments**.
