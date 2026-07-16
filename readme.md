# SpaceForge-XAI - Decoupled SPARTA Data-Generation Pipeline

> **Branch-specific README.** This document describes the **decoupled-sparta** branch only.
> The code and workflow here differ substantially from main (which couples SPARTA in-process
> via libsparta). On this branch the C++ simulator runs **decoupled** from SPARTA DSMC for
> much faster automated dataset generation.

---

## 1. What this branch is

SpaceForge-XAI simulates a **space-based MBE (molecular-beam epitaxy) wafer-growth platform**:
a C++/MPI harness ticks a set of spacecraft subsystems - solar array, battery, power bus,
effusion-cell heater, substrate heater, orbit model, deposition/growth monitors - while
executing **LLM-generated MBE scheduling recipes** (degas, oxide desorb, soak, nucleate,
growth, anneal, cooldown, idle).

On main, every run also drives a SPARTA DSMC gas simulation in-process, which dominates
runtime. On **this branch** the SPARTA coupling is **decoupled**:

- The build defaults to `ENABLE_SPARTA=OFF`, so the simulator links a lightweight
  **shim** ([Sim/src/SpartaBridgeShim.cpp](Sim/src/SpartaBridgeShim.cpp)) instead of libsparta.
- The shim launches SPARTA as an **external executable** (`$SPARTA_EXE`) - or, for pure
  C++ dataset generation, replaces it with **/bin/true** (`RUN_SPARTA=OFF`), so all
  scheduler/thermal/power code paths still execute but no DSMC runs.
- Result: a full 1350-tick (about 22.5 hours simulated) job finishes in seconds-to-minutes
  instead of hours, making batch generation of ML training data practical.

Each **tick = 1 simulated minute** (`dt = 60 s` internally; job files are specified in minutes).

---

## 2. Repository layout

```
spaceforge-xai/
|- CMakeLists.txt                # top-level build; ENABLE_SPARTA toggle (default OFF)
|- Sim/
|  |- CMakeLists.txt             # builds simcore static lib + sim executable
|  |- run.sh                     # build-and-run wrapper (see section 5)
|  |- run_orbit3.slurm           # MAIN data-generation Slurm script (see section 4)
|  |- run_orbit.slurm, run_orbit2.slurm   # older variants, kept for reference
|  |- include/                   # subsystem headers
|  |- src/
|     |- main.cpp                # entry point: arg parsing, job loading, tick loop
|     |- helpers.cpp             # CLI parsing, phase policies, recipe validation
|     |- SpartaBridge.cpp        # in-process SPARTA bridge (only if ENABLE_SPARTA=ON)
|     |- SpartaBridgeShim.cpp    # decoupled bridge: external spa_ / /bin/true
|     |- Logger.cpp              # per-subsystem CSV logging (honors OUTPUT_ROOT / SF_LOG_DIR)
|     |- SubstrateHeater.cpp, EffusionCell.cpp, HeaterBank.cpp,
|     |  SolarArray.cpp, Battery.cpp, PowerBus.cpp, orbit.cpp,
|     |  GrowthMonitor.cpp, DepositionMap.cpp, ...
|- active_jobs/                  # the 20 LLM-generated MBE recipes (V4_job1..20.txt)
|- active_jobs_testing/          # small recipes for smoke tests
|- input/                        # SPARTA decks + species data (used when RUN_SPARTA=ON)
|  |- in.wake_harness            # default wake deck for this branch
|  |- data/                      # ar/o .species and .vss files
|  |- surf/                      # cupola and 300 mm wafer geometry + viz scripts
|- scripts/                      # (cluster-side) config_row_to_args.py,
|                                #  submit_all_configs_serial.py  - see section 6 note
|- SpaceForge-xai Job tracker - Analytics.csv   # (cluster-side) config parameter table
|- tests/                        # minimal test target
```

> **Note:** `scripts/` and the Analytics CSV are referenced by the pipeline but are **not
> committed on this branch** - they exist only as untracked files in the run tree on the
> cluster (`/common/home/rvk22/spaceforge-xai-run2/`). Git never pushed them because they
> were never added/committed. Keep them alongside the repo root when deploying, or commit
> them from the cluster.

---

## 3. The data-generation pipeline at a glance

```
SpaceForge-xai Job tracker - Analytics.csv     (one row per Config: physics/power params)
        |
        |  scripts/config_row_to_args.py ConfigN
        |     prints: "--battery-capacity-wh 6000 --solar-efficiency 0.25 ..."
        v
Sim/run_orbit3.slurm ConfigN                   (one Slurm job per configuration)
        |
        |  loops over active_jobs/V4_job*.txt  (the 20 LLM-generated recipes,
        |  identical for every configuration - only the config params change)
        v
Sim/run.sh --nticks 1350 --job-file V4_jobM.txt <config args>
        |
        |  cmake + build, then: mpirun -np 8 sim --mode wake ...
        v
data/raw/ConfigN/V4_jobM/    (10 subsystem CSVs + logs + provenance snapshots)
```

Every configuration therefore produces **20 runs** (one per recipe), and each run produces
the same set of per-tick CSV time series. Because the 20 recipes are held constant across
configurations, the dataset isolates the effect of the configuration parameters.

---

## 4. Sim/run_orbit3.slurm - the main data-generation script

### Usage

```bash
# from the repo/run-tree root on the cluster
sbatch Sim/run_orbit3.slurm Config27
```

The single positional argument is the **config name** - it must match a row in
`SpaceForge-xai Job tracker - Analytics.csv`. Optional environment overrides:

```bash
MAX_JOBS=3 SIM_MODE=wake RUN_SPARTA=OFF sbatch Sim/run_orbit3.slurm Config27
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `MAX_JOBS` | `0` | Cap on how many recipe files to run. `0` = run **all** of them. |
| `SIM_MODE` | `wake` | Passed to the simulator as `MODE` (`wake`, `power`, `dual`, `legacy`). |
| `RUN_SPARTA` | `OFF` | `OFF` = pure C++ dataset mode (SPARTA replaced by /bin/true). `ON` = launch the real external SPARTA binary. |

Slurm resources: 8 tasks x 1 CPU, 32 GB, 100 h wall time. stdout/stderr go to
`logs/sf_queue_<jobid>.{out,err}`.

### What it does, step by step

1. **Validates the environment** - the tracker CSV, `scripts/config_row_to_args.py`,
   and the `active_jobs/` directory must all exist; otherwise it exits with a clear error.
2. **Resolves config parameters** - runs
   `python3 scripts/config_row_to_args.py "<tracker.csv>" "<ConfigName>"`, which looks up
   the named config's row in the Analytics CSV and prints it as a string of simulator CLI
   flags (e.g. `--battery-capacity-wh 6000 --solar-efficiency 0.25 ...`). The script
   evals that into an argument array used for every job in the config.
3. **Prepares the config output folder** - `data/raw/<ConfigName>/`, and writes provenance:
   `runtime_args_config_level.txt` (the resolved flags) and `analytics_snapshot.csv`
   (a copy of the tracker CSV at run time).
4. **Discovers the recipes** - every `active_jobs/V4_job*.txt`, sorted numerically
   (V4_job1 ... V4_job20). These are the **LLM-generated MBE scheduling recipes**
   (section 7). They are the same for every configuration.
5. **For each recipe (job)**:
   - Creates `data/raw/<ConfigName>/<V4_jobM>/` and scrubs any stale CSVs/logs from a
     previous attempt (both in the job folder and in `Sim/`).
   - Writes per-job provenance: `runtime_args.txt`, `analytics_snapshot.csv`, a copy of the
     recipe file, and `params.csv` (the config flags parsed into a parameter,value table
     by an inline Python snippet).
   - Invokes **run.sh** (see section 5) from `Sim/` with a fully specified environment:

     ```bash
     RUN_ID="<ConfigName>_<V4_jobM>" \
     OUTPUT_ROOT="<job output dir>" SF_LOG_DIR="<job output dir>" \
     REQUIRE_OUTPUT_ROOT=ON \
     BUILD_DIR="build_work/<slurmid>_<runid>" \
     MODE="$SIM_MODE" RUN_SPARTA="$RUN_SPARTA" ENABLE_SPARTA=OFF \
     GPU=OFF NP=8 J=8 \
     ./run.sh --nticks 1350 --job-file "<abs path to recipe>" <config args...> \
         > "<job dir>/runtime_stdout_stderr.log" 2>&1
     ```

     Note `--nticks 1350` - 1350 simulated minutes (22.5 hours), enough to cover the
     longest recipes plus queue/cooldown behavior.
   - **Sweeps stragglers**: any CSVs or `sim_debug_*` / `sim_rank_progress_*` logs that
     landed in `Sim/` are moved into the job folder; `run_spa.log` is deleted.
   - **Guards against logger regressions**: if a loose `data/raw/<ConfigName>_<V4_jobM>/`
     folder appears, the script aborts with exit code **11** - that means Logger.cpp is
     appending RUN_ID instead of honoring OUTPUT_ROOT.
   - **Verifies outputs**: the job folder must contain all ten non-empty subsystem CSVs -
     Battery.csv, EffusionCell.csv, HeaterBank.csv, Orbit.csv, PowerBus.csv,
     ProcessState.csv, ScheduleState.csv, SimulationEngine.csv, SolarArray.csv,
     substrate.csv. Missing/empty CSVs fail the whole config with exit code **10**.
   - Writes `job_status.txt` (config, job id, exit code, missing-CSV report) for auditing.
   - A non-zero simulator exit code aborts the config immediately with that code.
6. Prints a completion banner once all requested jobs for the config are done.

> **Hardcoded path:** `ROOT_DIR="/common/home/rvk22/spaceforge-xai-run2"` at the top of the
> script. Edit this (and re-check TRACKER_CSV / CONFIG_ARGS_SCRIPT) when deploying to a
> different account or directory.

### Output layout per config

```
data/raw/Config27/
|- runtime_args_config_level.txt
|- analytics_snapshot.csv
|- V4_job1/
|  |- Battery.csv  EffusionCell.csv  HeaterBank.csv  Orbit.csv  PowerBus.csv
|  |- ProcessState.csv  ScheduleState.csv  SimulationEngine.csv  SolarArray.csv  substrate.csv
|  |- params.csv  runtime_args.txt  analytics_snapshot.csv  V4_job1.txt
|  |- runtime_stdout_stderr.log  sim_debug_*.log  sim_rank_progress_*.log
|  |- job_status.txt
|- V4_job2/ ...
|- V4_job20/
```

---

## 5. Sim/run.sh - build-and-run wrapper

run.sh is the single entry point for building and executing the simulator, locally or
under Slurm. It is entirely **environment-variable driven**, with CLI arguments passed
through verbatim to the sim binary.

### What it does

1. **Resolves identity and output**: sanitizes RUN_ID; determines OUTPUT_ROOT
   (defaults to `data/raw/<RUN_ID>` unless `REQUIRE_OUTPUT_ROOT=ON`, in which case it
   refuses to run without an explicit OUTPUT_ROOT - this is how the Slurm script prevents
   accidental loose folders).
2. **Chooses a build dir**: BUILD_DIR if given, else `build_work/<slurmid>_<runid>` under
   Slurm, else `Sim/build`. The build dir is **wiped and reconfigured every run** so source
   edits always take effect.
3. **Selects the SPARTA path** based on GPU:
   - `GPU=ON` uses `~/opt/sparta/build-gpu/src/spa_` with Kokkos/CUDA args (`-k on g 1 -sf kk`)
   - `GPU=OFF` uses `~/opt/sparta/src/spa_` (CPU)
4. **Configures the SPARTA execution mode** (the key decoupling switch):
   - `RUN_SPARTA=OFF` exports `SPARTA_EXE=/bin/true`. The simulator's wake-scheduler code
     paths all run, but the "SPARTA launch" is a no-op. This is **C++-only dataset mode**.
   - `RUN_SPARTA=ON` exports the real SPARTA_EXE plus SPARTA_EXTRA_ARGS containing
     orbit/environment `-var` knobs read from env: PTORR_TARGET, PCUP_TORR,
     CUP_BASE_SCALE, CUP_AMP_SCALE, CUP_PHASE0, and runID.
   - Independently, ENABLE_SPARTA (default OFF) controls **build linkage**: OFF links
     the external-executable shim; ON compiles/links libsparta in-process (the main
     branch behavior).
5. **Builds**: `cmake -S <repo root> -B $BUILD_DIR -DENABLE_SPARTA=... -DCMAKE_BUILD_TYPE=Release`
   then `cmake --build -j $J`. The binary is `$BUILD_DIR/Sim/sim`.
6. **Runs from OUTPUT_ROOT** (so all relative outputs land in the job folder), headless:

   ```bash
   env -u DISPLAY -u XAUTHORITY mpirun -np $NP $SIM_EXE \
     --mode $MODE --wake-deck <abs deck path> --input-subdir <abs input dir> "$@"
   ```

7. Prints a summary of every knob at startup and lists the CSVs found in OUTPUT_ROOT
   afterwards.

### Environment knobs

| Variable | Default | Purpose |
| --- | --- | --- |
| `RUN_ID` | `run_default` | Run identifier used in log filenames and SPARTA runID. |
| `OUTPUT_ROOT` | `data/raw/<RUN_ID>` | Where the run executes and CSVs land. |
| `REQUIRE_OUTPUT_ROOT` | `OFF` | `ON` = error out if OUTPUT_ROOT not explicitly set. |
| `BUILD_DIR` | auto | CMake build directory (recreated each run). |
| `GPU` | `OFF` | Select GPU (Kokkos/CUDA) vs CPU SPARTA executable/args. |
| `ENABLE_SPARTA` | `OFF` | Build-time: link libsparta in-process (ON) vs external shim (OFF). |
| `RUN_SPARTA` | `OFF` | Run-time: real external SPARTA (ON) vs /bin/true stub (OFF). |
| `SPARTA_DIR` / `SPARTA_EXE` | `~/opt/sparta/...` | Override SPARTA locations. |
| `NP` | `1` | MPI ranks for the simulator. |
| `J` | `8` | Parallel build jobs. |
| `MODE` | `wake` | Simulator mode. |
| `WAKE_DECK` | `in.wake` | Wake deck (Slurm/main default on this branch: in.wake_harness). |
| `INPUT_SUBDIR` | `<repo>/input` | Absolute input dir passed to the simulator. |
| `PTORR_TARGET`, `PCUP_TORR`, `CUP_BASE_SCALE`, `CUP_AMP_SCALE`, `CUP_PHASE0` | see script | Orbit/pressure knobs forwarded to the wake deck when RUN_SPARTA=ON. |
| `CC` / `CXX` | `mpicc` / `mpicxx` | Compilers. |

### Example: single manual run (no Slurm)

```bash
cd Sim
RUN_ID=smoke_test MODE=wake RUN_SPARTA=OFF NP=8 \
./run.sh --nticks 1350 --job-file "$PWD/../active_jobs/V4_job1.txt" \
         --config-name Config1
# outputs appear in ../data/raw/smoke_test/
```

---

## 6. Batch generation of all configurations

To generate the full dataset - every configuration in the tracker CSV, run serially,
each with all 20 recipes - use:

```bash
MAX_JOBS=0 SIM_MODE=wake RUN_SPARTA=OFF python3 scripts/submit_all_configs_serial.py
```

- `scripts/submit_all_configs_serial.py` reads the config rows from
  `SpaceForge-xai Job tracker - Analytics.csv` and drives one run_orbit3.slurm execution
  per configuration, **serially** (one config finishes before the next starts), forwarding
  the MAX_JOBS / SIM_MODE / RUN_SPARTA environment to each.
- `MAX_JOBS=0` means every config runs **all 20 recipes**; set e.g. `MAX_JOBS=1` for a fast
  end-to-end smoke pass over all configs.
- `RUN_SPARTA=OFF` keeps everything in fast C++-only mode - the standard setting for
  dataset generation on this branch.

**The invariant that makes the dataset useful:** the 20 LLM-generated recipes in
`active_jobs/` are held **constant across all configurations**; only the configuration
parameters (battery sizing, solar input/efficiency, thermal coefficients, failure limits,
etc.) change from row to row of the Analytics CSV. Each config's row is translated into
simulator CLI flags by `scripts/config_row_to_args.py` and snapshotted into every job
folder (`params.csv`, `runtime_args.txt`, `analytics_snapshot.csv`) for full provenance.

> Like the tracker CSV, the `scripts/` directory is part of the cluster run tree and is not
> committed on this branch.

---

## 7. The LLM-generated MBE recipes (active_jobs/)

Each `V4_jobN.txt` is a whitespace-separated schedule; comments start with `#`.

```
# start_tick end_tick wafer_flux_cm2s heater_W mbe_on substrate_on phase_code substrate_target_K
0    180   0.0e0     1800   0   0   SOURCE_DEGAS   300.0
180  210   0.0e0     0      0   1   OXIDE_DESORB   893.0
225  240   8.0e12    1700   1   1   NUCLEATE       803.0
240  330   2.4e13    2200   1   1   GROWTH         873.0
...
```

| Column | Meaning |
| --- | --- |
| `start_tick`, `end_tick` | Phase window in **minutes** of sim time. |
| `wafer_flux_cm2s` | Requested growth flux (cm^-2 s^-1) when mbe_on=1; also used as a thermal anchor for source phases. |
| `heater_W` | Effusion heater **power cap** in watts (not a temperature setpoint). |
| `mbe_on` | Beam shutter: 1 only in growth-like phases. |
| `substrate_on` | Substrate heater control enabled. |
| `phase_code` | One of IDLE, SOURCE_DEGAS, OXIDE_DESORB, SOAK, NUCLEATE, GROWTH, ANNEAL, COOLDOWN. |
| `substrate_target_K` | Substrate temperature setpoint (K). |

Rules enforced by the loader ([helpers.cpp](Sim/src/helpers.cpp) validateJob): growth-like
phases (NUCLEATE, GROWTH) require mbe_on=1 and positive flux; all other timed phases
require mbe_on=0; OXIDE_DESORB/SOAK/ANNEAL/COOLDOWN require substrate_on=1;
IDLE requires beam off and zero flux. Malformed lines are skipped with a logged warning.
A legacy 4-column format (start end flux heater_W) is still accepted.

Source-side behavior per phase is derived by a scheduler policy (deriveSourcePhasePolicy):
e.g. SOURCE_DEGAS holds the source about 100 K above the flux-derived growth target, SOAK
holds about 25 K below it, ANNEAL backs off about 150 K, and growth phases use
targetTempForFlux() - a monotonic log-flux to 1100-1500 K mapping.

Recipes may contain **multiple back-to-back recipe blocks** (e.g. "Recipe A" then a
lower-temperature "Recipe B"), including overlapping windows to exercise queueing and
cooldown interactions.

---

## 8. Simulator CLI reference (sim)

Run `sim --help` for the authoritative list. Runtime defaults equal **Config 1 (row 10)**
of the Analytics CSV, so a bare run is still a valid experiment.

**Core flags**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `dual` | `wake` (standard here), `power` (no SPARTA paths at all), `dual` (alias of wake), `legacy`. |
| `--wake-deck` | `in.wake_harness` | Wake deck (absolute path recommended). |
| `--input-subdir` | `input` | Input directory. |
| `--job-file` | `V4_job1.txt` | Recipe file; absolute path, or relative to input dir. |
| `--nticks` | 500 | Engine ticks (minutes). Slurm uses **1350**. |
| `--dt` | 60.0 | Seconds per tick. |
| `--config-name` | `Config 1` | Label recorded in logs/CSVs. |
| `--couple-every` / `--sparta-block` | 10 / 200 | SPARTA coupling cadence (relevant when SPARTA is live). |

**Configuration parameters** (these are what the Analytics CSV varies per config)

| Flag | Default | |
| --- | --- | --- |
| `--battery-capacity-wh` | 6000 | Battery capacity. |
| `--battery-start-charge-wh` | 3000 | Initial charge. |
| `--battery-max-discharge-w` / `--battery-max-charge-w` | 4000 / 3500 | Power limits. |
| `--solar-base-input-w` | 30000 | Insolation input. |
| `--solar-efficiency` | 0.25 | Array efficiency. |
| `--effusion-h-wk` | 0.8 | Effusion cell thermal conductance (W/K). |
| `--effusion-c-j` | 800 | Effusion cell heat capacity (J/K). |
| `--effusion-night-ambient-k` / `--effusion-day-ambient-k` | 250 / 325 | Orbit-dependent ambient (signed warming/cooling). |
| `--substrate-c-j` | 1500 | Substrate heat capacity (J/K). |
| `--substrate-eps` | 0.8 | Substrate emissivity. |
| `--substrate-max-power-w` | 3000 | Substrate heater cap. |
| `--substrate-ready-band-k` | 5.0 | Temperature band counted as "at target". |
| `--substrate-fail-limit-ticks` | 20 | Consecutive out-of-band ticks before failure. |
| `--effusion-underflux-limit-ticks` / `--effusion-undertemp-limit-ticks` | 20 / 20 | Source failure streak caps. |
| `--effusion-min-flux-fraction` | 0.9 | Minimum acceptable flux fraction. |
| `--effusion-temp-tolerance-fraction` | 0.85 | Source temperature readiness threshold. |
| `--heater-bank-max-draw-w` | 5000 | Heater bank power cap. |

---

## 9. Running with real SPARTA (optional)

For runs that need actual DSMC wake physics:

```bash
# CPU SPARTA
RUN_SPARTA=ON GPU=OFF sbatch Sim/run_orbit3.slurm Config27

# GPU (Kokkos/CUDA) SPARTA
RUN_SPARTA=ON GPU=ON  sbatch Sim/run_orbit3.slurm Config27
```

The shim launches `mpirun -np $SPARTA_NP $SPARTA_EXE -in in.wake_harness <SPARTA_EXTRA_ARGS>`
from the deck directory, logging to `run_spa.log`. Orbit/pressure variables
(pTorrTarget, Pcup_Torr, cup_base_scale, cup_amp_scale, phase0) are injected as
SPARTA `-var`s. See [Sim/readmeSim.md](Sim/readmeSim.md) for building SPARTA itself
(CPU and GPU/Kokkos instructions).

To restore the main-style **in-process** coupling instead, build with
`ENABLE_SPARTA=ON` and a valid SPARTA_DIR containing `libsparta*.a`.

---

## 10. Outputs

Every run writes ten per-tick CSV time series into OUTPUT_ROOT:

Battery.csv, EffusionCell.csv, HeaterBank.csv, Orbit.csv, PowerBus.csv,
ProcessState.csv, ScheduleState.csv, SimulationEngine.csv, SolarArray.csv,
substrate.csv

plus diagnostics: `sim_debug_<RUN_ID>_<mode>.log` (rank-0 event log),
`sim_rank_progress_r<K>_*.log` (per-rank progress, for MPI hang debugging), and
`params.csv` (runtime parameter snapshot written by the simulator itself).
Logger.cpp and GrowthMonitor.cpp honor OUTPUT_ROOT / SF_LOG_DIR directly - they
must **not** append RUN_ID (the Slurm script aborts with exit 11 if they do).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ERROR: missing config name` | run_orbit3.slurm needs a positional config name: `sbatch Sim/run_orbit3.slurm Config27`. |
| `ERROR: tracker csv not found` | Put `SpaceForge-xai Job tracker - Analytics.csv` at the run-tree root, or fix TRACKER_CSV in the script. |
| `failed to load parameters for config` | The config name doesn't match a row in the CSV, or `scripts/config_row_to_args.py` is missing. |
| Exit code **10** | Run finished but at least one expected subsystem CSV is missing/empty - check `runtime_stdout_stderr.log` in the job folder. |
| Exit code **11** | Logger regression: outputs went to `data/raw/<Config>_<job>/` instead of OUTPUT_ROOT. |
| `[fatal] Could not open job file` | `--job-file` path wrong; relative paths resolve under `--input-subdir`, so prefer absolute paths (the Slurm script already passes absolute paths). |
| `[run.sh] ERROR: OUTPUT_ROOT is required` | `REQUIRE_OUTPUT_ROOT=ON` without an explicit OUTPUT_ROOT. |
| Simulator binary missing | CMake/build failure - scroll up in `runtime_stdout_stderr.log`; the build dir is recreated every run so stale caches are not the cause. |
| X11 "Authorization required" noise | Harmless; runs are already headless (`env -u DISPLAY -u XAUTHORITY`). |

---

## License and acknowledgments

- SPARTA is (c) Sandia National Laboratories; see SPARTA's own license for terms.
- This branch exists to generate explainable-AI training datasets from scheduler-driven
  MBE simulations; it is not a chemistry-complete MBE digital twin.
