# SpaceForge-XAI - Coupled SPARTA + C++ Harness (main branch)

> **Branch-specific README.** This document describes the **main** branch, where the C++
> simulator and SPARTA DSMC run **coupled in-process** (the simulator links libsparta and
> advances SPARTA every tick). For the fast, decoupled dataset-generation pipeline
> (SPARTA replaced by an external binary or a no-op), see the **decoupled-sparta** branch
> and its README.

---

## 1. What this is

SpaceForge-XAI simulates a **space-based MBE (molecular-beam epitaxy) wafer-growth
platform** in low Earth orbit. A C++/MPI harness ticks the spacecraft subsystems - solar
array, battery, power bus, effusion-cell heater, substrate heater, orbit model - once per
**simulated minute**, while executing **LLM-generated MBE scheduling recipes** (degas,
oxide desorb, soak, nucleate, growth, anneal, cooldown, idle).

On this branch every run is **physics-coupled**:

- The build uses `ENABLE_SPARTA=ON`, so [Sim/src/SpartaBridge.cpp](Sim/src/SpartaBridge.cpp)
  links directly against **libsparta** (SPARTA's C library API) and keeps one persistent
  SPARTA instance alive for the whole run.
- The wake deck [input/in.wake_harness](input/in.wake_harness) models the **LEO wake +
  wake-shield facility (WSF) + 300 mm wafer + MBE orifice** geometry with an
  O/N/N2/O2 freestream at 7500 m/s plus H2O outgassing from the shield.
- Every engine tick, the harness hands the current wafer flux and beam state to SPARTA
  through `input/params.inc`, then advances SPARTA by a block of DSMC steps. SPARTA in
  turn appends pressure-probe and residual-gas CSV rows for that tick.

The cost of full coupling is runtime: a 1350-tick job advances SPARTA by
1350 x 2500 = 3.375 million DSMC steps, so runs take hours (this is exactly what the
decoupled-sparta branch exists to avoid when only the C++-side dataset is needed).

---

## 2. Repository layout

```
spaceforge-xai/
|- CMakeLists.txt                # top-level build; ENABLE_SPARTA option, libsparta import
|- Sim/
|  |- CMakeLists.txt             # simcore static lib + sim executable
|  |- run.sh                     # build-and-run wrapper (see section 5)
|  |- run_orbit3.slurm           # MAIN Slurm queue script (see section 4)
|  |- run_orbit.slurm, run_orbit2.slurm   # older variants
|  |- readmeSim.md               # how to build SPARTA itself (CPU and GPU/Kokkos)
|  |- include/                   # subsystem headers
|  |- src/
|     |- main.cpp                # entry point: modes, job loading, tick loop, coupling
|     |- helpers.cpp             # CLI parsing, phase policies, recipe validation
|     |- SpartaBridge.cpp        # in-process libsparta wrapper (this branch's default)
|     |- SpartaBridgeShim.cpp    # external-binary fallback (used if ENABLE_SPARTA=OFF)
|     |- WakeChamber.cpp         # persistent SPARTA facade: init once, advance in blocks
|     |- Logger.cpp              # per-subsystem CSV logging to data/raw/<RUN_ID>/
|     |- SubstrateHeater.cpp, EffusionCell.cpp, HeaterBank.cpp,
|     |  SolarArray.cpp, Battery.cpp, PowerBus.cpp, orbit.cpp,
|     |  GrowthMonitor.cpp, DepositionMap.cpp, ...
|- active_jobs/                  # the 20 LLM-generated MBE recipes (V4_job1..20.txt)
|- active_jobs_testing/          # small recipes for smoke tests
|- input/
|  |- in.wake_harness            # MAIN wake deck: WSF shield + wafer + MBE orifice
|  |- in.wake_harness_cupola     # variant: cupola pencil-beam orifice geometry
|  |- params.inc                 # handshake file written by the harness every tick
|  |- data/                      # o.species / o.vss (O,N,N2,O2,H2O), ar.species / ar.vss
|  |- surf/                      # wsf.surf, wafer_300mm.surf, mbe_orifice_10mm.surf,
|                                #  cupola geometry + Python viz/repair scripts
|- tests/                        # minimal test target
```

---

## 3. How the coupling works

1. **Startup (wake mode).** Rank 0 loads the recipe given by `--job-file`, then all ranks
   construct a `WakeChamber` on `MPI_COMM_WORLD`. `WakeChamber::init` opens libsparta,
   changes into `input/`, and reads `in.wake_harness` once. The deck stays resident for
   the whole run.
2. **The handshake file.** The deck begins with `include params.inc`. The harness writes
   `input/params.inc` (rank 0 only) with two variables - `Fwafer_cm2s` (requested wafer
   flux) and `mbe_active` (beam on/off) - at startup, whenever the scheduler changes
   phase, and on abort paths. When values change, the chamber is marked dirty so the deck
   is cleared and re-read with the new values.
3. **Advancing SPARTA.** Every `--couple-every` engine ticks, all ranks call
   `wake.runIfDirtyOrAdvanceCollective(--sparta-block)`, which either re-reads the deck
   (if dirty) or issues `run N` without re-reading. The Slurm script uses
   `--couple-every 1 --sparta-block 2500`, and the deck defines `variable block equal 2500`
   with `tick = floor(step/block)` - so **one engine tick (1 simulated minute) equals one
   deck tick equals 2500 DSMC steps**. If you change `--sparta-block`, change the deck's
   `block` variable to match, or the SPARTA-side tick counter will drift.
4. **SPARTA-side output.** Each deck tick, SPARTA appends one row to
   `data/raw/<runID>/wake_4probes.csv` (front/wake/freestream/gap pressures in Torr,
   cupola outgassing scale and emission rate, logged wafer flux, beam state) and one row
   to `data/raw/<runID>/residualGasAnalyzer.csv` (per-species wake partial pressures:
   O, N, H, He, N2, O2, Ar, H2O, plus the RGA sum). The `runID` deck variable is injected
   by run.sh as `-var runID <RUN_ID>`, so SPARTA CSVs land in the same folder as the
   C++ CSVs.
5. **Orbit knobs.** run.sh also injects `-var pTorrTarget` (freestream ambient pressure),
   `-var Pcup_Torr` (target wake-side outgassing addition), and sinusoidal outgassing
   scale parameters (`cup_base_scale`, `cup_amp_scale`, `phase0`) so the wake environment
   varies over the orbit.

Modes other than `wake`: `power` runs the C++ power/thermal harness with no SPARTA at
all; `dual` is currently an alias of wake; `legacy` is the old single-instance path.

---

## 4. Sim/run_orbit3.slurm - the batch queue script

### Usage

```bash
# from the run-tree root on the cluster
sbatch Sim/run_orbit3.slurm
```

No arguments. Slurm resources: 8 tasks x 1 CPU, 32 GB, 100 h wall time; stdout/stderr in
`logs/sf_queue_<jobid>.{out,err}`.

### What it does

The main-branch script is a simple sequential queue (much simpler than the
decoupled-sparta version - no config CSV, no output verification):

1. `cd /common/home/rvk22/spaceforge-xai-run5` (**hardcoded** - edit for your account).
2. Loops over **every** `active_jobs/*.txt` recipe file, in shell glob order.
3. For each recipe, runs from `Sim/`:

   ```bash
   RUN_ID="<recipe name>" \
   MODE=wake \
   ENABLE_SPARTA=ON \
   GPU=OFF \
   WAKE_DECK=in.wake_harness \
   NP=8 \
   ./run.sh --nticks 1350 --couple-every 1 --sparta-block 2500 \
            --job-file "../active_jobs/<recipe>.txt"
   ```

   - `ENABLE_SPARTA=ON` - build and link libsparta in-process (the defining setting of
     this branch).
   - `--nticks 1350` - 1350 simulated minutes (22.5 hours), covering the longest recipes
     plus queue/cooldown behavior.
   - `RUN_ID` is the recipe name, so outputs land in `data/raw/V4_jobN/`.
   - The job file path is passed **relative**: the simulator resolves `--job-file` as
     `<input dir>/<value>`, and since run.sh passes the absolute `input/` dir, the
     `../active_jobs/...` prefix walks from `input/` back up to the repo root. Keep that
     `../` prefix if you add your own recipes.

Each recipe runs to completion (or failure) before the next starts. There is no
per-config parameter sweep on this branch - the physics/power constants are compiled-in
defaults, and the dataset dimension is the recipe, not the configuration.

---

## 5. Sim/run.sh - build-and-run wrapper

Environment-variable driven; CLI args pass through verbatim to the sim binary.

### What it does

1. Wipes and reconfigures `BUILD_DIR` (default `Sim/build`) every invocation, so source
   edits always take effect:
   `cmake -S <repo root> -B $BUILD_DIR -DSPARTA_DIR=... -DENABLE_SPARTA=... -DCMAKE_BUILD_TYPE=Release`
   then `cmake --build -j $J`.
2. Selects SPARTA locations from `GPU`:
   - `GPU=OFF` uses `~/opt/sparta/src` (CPU build of libsparta / spa_)
   - `GPU=ON` uses `~/opt/sparta/build-gpu/src` and adds Kokkos args (`-k on g 1 -sf kk`)
3. Builds `SPARTA_EXTRA_ARGS` with the orbit `-var` knobs (pTorrTarget, Pcup_Torr,
   cup_base_scale, cup_amp_scale, phase0, runID) unless the caller already set it.
   `SpartaBridge` reads `SPARTA_EXTRA_ARGS` from the environment when opening libsparta.
4. Creates `data/raw/<RUN_ID>/` and runs headless from the build directory:

   ```bash
   env -u DISPLAY -u XAUTHORITY mpirun -np $NP ./Sim/sim \
     --mode $MODE --wake-deck $WAKE_DECK --input-subdir <abs input dir> "$@"
   ```

### Environment knobs

| Variable | Default | Purpose |
| --- | --- | --- |
| `RUN_ID` | `run_default` | Output subfolder name under `data/raw/` (also SPARTA's runID). |
| `MODE` | `wake` | Simulator mode (`wake`, `power`, `dual`, `legacy`). |
| `WAKE_DECK` | `in.wake` | Deck filename in `input/` (Slurm uses `in.wake_harness`). |
| `ENABLE_SPARTA` | `OFF` | `ON` = link libsparta in-process (what the Slurm script uses). `OFF` = external-binary shim fallback. |
| `GPU` | `OFF` | CPU vs GPU (Kokkos/CUDA) SPARTA paths and args. |
| `SPARTA_DIR` / `SPARTA_EXE` | `~/opt/sparta/...` | Override SPARTA locations. |
| `NP` | `1` | MPI ranks (Slurm uses 8). |
| `J` | `8` | Parallel build jobs. |
| `BUILD_DIR` | `Sim/build` | CMake build dir (recreated each run). |
| `INPUT_SUBDIR` | `<repo>/input` | Absolute input dir. |
| `PTORR_TARGET` | `1.0e-7` | Freestream ambient pressure [Torr]. |
| `PCUP_TORR` | `9.0e-9` | Target wake-side outgassing addition [Torr]. |
| `CUP_BASE_SCALE` / `CUP_AMP_SCALE` / `CUP_PHASE0` | `1.0` / `0.50` / `0.0` | Sinusoidal orbital outgassing scale. |
| `CC` / `CXX` | `mpicc` / `mpicxx` | Compilers. |

### Example: single manual run

```bash
cd Sim
RUN_ID=test_run1 MODE=wake ENABLE_SPARTA=ON GPU=OFF WAKE_DECK=in.wake_harness NP=8 \
./run.sh --nticks 1350 --couple-every 1 --sparta-block 2500 \
         --job-file ../active_jobs/V4_job1.txt
# outputs appear in ../data/raw/test_run1/
```

Prerequisite: a built SPARTA with `libsparta_mpi.a` (or `libsparta.a`) under
`$SPARTA_DIR`. See [Sim/readmeSim.md](Sim/readmeSim.md) for SPARTA build instructions
(CPU and GPU/Kokkos). If the library is missing, CMake fails with
"Couldn't find libsparta*.a".

---

## 6. The LLM-generated MBE recipes (active_jobs/)

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
| `wafer_flux_cm2s` | Requested growth flux (cm^-2 s^-1) when mbe_on=1; also a thermal anchor for source phases. |
| `heater_W` | Effusion heater **power cap** in watts (not a temperature setpoint). |
| `mbe_on` | Beam shutter: 1 only in growth-like phases. |
| `substrate_on` | Substrate heater control enabled. |
| `phase_code` | One of IDLE, SOURCE_DEGAS, OXIDE_DESORB, SOAK, NUCLEATE, GROWTH, ANNEAL, COOLDOWN. |
| `substrate_target_K` | Substrate temperature setpoint (K). |

Validation rules ([helpers.cpp](Sim/src/helpers.cpp) validateJob): growth-like phases
(NUCLEATE, GROWTH) require mbe_on=1 and positive flux; other timed phases require
mbe_on=0; OXIDE_DESORB/SOAK/ANNEAL/COOLDOWN require substrate_on=1; IDLE requires beam
off and zero flux. Malformed lines are skipped with a warning. A legacy 4-column format
(start end flux heater_W) is still accepted. If the job file is missing, the run
continues with default heater/flux values (it does not abort on this branch).

Source-side behavior per phase comes from the scheduler policy (deriveSourcePhasePolicy):
SOURCE_DEGAS holds the source about 100 K above the flux-derived growth target, SOAK
about 25 K below it, ANNEAL backs off about 150 K, and growth phases use
targetTempForFlux() - a monotonic log-flux to 1100-1500 K mapping. During growth, the
requested flux and beam state flow into SPARTA through params.inc, so the DSMC wake
responds to the recipe in real time.

---

## 7. Simulator CLI reference (sim)

Run `sim --help` for the authoritative list.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `dual` | `wake` (standard), `power` (no SPARTA), `dual` (alias of wake), `legacy`. |
| `--wake-deck` | `in.wake_harness` | Deck filename, resolved in the SPARTA working dir (`input/`). |
| `--input-subdir` | `input` | Input directory (run.sh passes it absolute). |
| `--job-file` | `V4_job1.txt` | Recipe file, resolved as `<input dir>/<value>` (relative only on this branch - use `../active_jobs/...`). |
| `--nticks` | 500 | Engine ticks (simulated minutes). Slurm uses 1350. |
| `--dt` | 60.0 | Seconds per tick. |
| `--couple-every` | 10 | Advance SPARTA every N engine ticks (Slurm uses 1). |
| `--sparta-block` | 200 | DSMC steps per advance (Slurm uses 2500; must match the deck's `block` variable). |
| `--split` | size/2 | Rank split for the historical dual mode. |

Unlike the decoupled-sparta branch, there are **no per-experiment physics flags**
(battery capacity, solar efficiency, thermal constants, etc.) - those values are
compiled-in defaults in the subsystem classes on this branch.

---

## 8. The wake decks (input/)

### in.wake_harness (used by the Slurm script)

- 3D LEO wake at 400 km: freestream mixture O/N/N2/O2 (pymsis-derived fractions,
  normalized) drifting at 7500 m/s, Tgas 800 K, open boundaries, timestep 1.0e-5 s.
- Geometry via `read_surf`: `surf/wsf.surf` (wake-shield facility), `surf/wafer_300mm.surf`
  (300 mm wafer behind the shield), `surf/mbe_orifice_10mm.surf` (MBE source orifice).
- H2O outgassing is emitted from the WSF wake face, scaled sinusoidally over the orbit
  (`cup_base_scale`, `cup_amp_scale`, `phase0`) toward a target wake addition of
  `Pcup_Torr`.
- Harness-driven variables `Fwafer_cm2s` and `mbe_active` arrive via `include params.inc`.
- Outputs per deck tick (2500 steps): `wake_4probes.csv` (front/wake/free/gap pressures
  and emission state) and `residualGasAnalyzer.csv` (per-species wake partial pressures),
  both appended under `data/raw/<runID>/`.

### in.wake_harness_cupola (variant)

Same harness-driven pattern, but with the cupola pencil-beam MBE orifice geometry and
the orbit scaling internal to the deck. Select it with `WAKE_DECK=in.wake_harness_cupola`.

---

## 9. Outputs

All outputs collect under `data/raw/<RUN_ID>/`:

- **C++ subsystem CSVs** (one row per tick, written via Logger): Battery.csv,
  EffusionCell.csv, HeaterBank.csv, Orbit.csv, PowerBus.csv, ProcessState.csv,
  ScheduleState.csv, ScheduleStateText.csv, SimulationEngine.csv, SolarArray.csv,
  substrate.csv, WakeChamber.csv, plus an Events log.
- **SPARTA CSVs** (one row per deck tick): wake_4probes.csv, residualGasAnalyzer.csv.
- **Diagnostics**: `sim_debug_<RUN_ID>_<mode>.log` (rank-0 event log) and
  `sim_rank_progress_r<K>_*.log` (per-rank progress for MPI hang debugging) in the
  directory the simulator ran from; `log.capi` / SPARTA logs in `input/`.

Logger resolves its base directory as `SF_LOG_DIR` if set, else
`<repo>/data/raw`, and appends `RUN_ID` as a subfolder.

---

## 10. Build notes

- Top-level [CMakeLists.txt](CMakeLists.txt): `-DENABLE_SPARTA=ON` requires
  `-DSPARTA_DIR` (or env `SPARTA_DIR`) pointing at a SPARTA tree containing
  `libsparta_mpi.a` or `libsparta.a`. With `ENABLE_SPARTA=OFF` a stub target is used and
  the external-binary shim is compiled instead.
- `PROJECT_SOURCE_DIR` is baked in so deck-relative paths (`data/...`, `surf/...`)
  resolve when SPARTA changes into `input/`.
- Requires MPI (OpenMPI tested), CMake 3.20+, C++17.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| CMake: `Couldn't find libsparta*.a` | Build SPARTA first (see [Sim/readmeSim.md](Sim/readmeSim.md)) or fix SPARTA_DIR. |
| `[info] No jobs.txt found at ...` | `--job-file` path wrong - it resolves under the input dir, so recipes need the `../active_jobs/` prefix. The run continues with defaults, so check this line in the log. |
| SPARTA tick counter drifts from harness ticks | `--sparta-block` no longer matches the deck's `variable block equal 2500`. Keep them equal. |
| Deck includes not found | SPARTA runs with cwd `input/`; make sure `params.inc` exists (the harness writes it at startup) and `data/` / `surf/` paths are intact. |
| Runs take hours | Expected on this branch: full DSMC coupling. Use the decoupled-sparta branch for fast C++-only dataset generation. |
| X11 "Authorization required" noise | Harmless; runs are headless (`env -u DISPLAY -u XAUTHORITY`). |
| Too much console spam | Increase `stats` interval in the deck. |

---

## License and acknowledgments

- SPARTA is (c) Sandia National Laboratories; see SPARTA's own license for terms.
- This project is a research scaffold for coupling DSMC wake physics to spacecraft
  power/thermal scheduling; it is not a chemistry-complete MBE digital twin.
