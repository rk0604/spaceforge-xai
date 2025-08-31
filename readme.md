# SpaceForge-XAI — Dual SPARTA + Power Subsystems (MPI)

This repo is a small but realistic scaffold for coupling **SPARTA** DSMC gas simulations to a toy **electrical/power model** (solar array, battery, heater bank) under **MPI**. It demonstrates:
- Running **two persistent SPARTA instances at once** (Wake + Effusion) by splitting `MPI_COMM_WORLD`.
- Periodically **advancing SPARTA** while the power subsystems tick on their own cadence.
- Clean separation: a thin `SpartaBridge` C-API wrapper and a higher‑level `WakeChamber` façade that keeps a SPARTA instance alive between advances.

> Works headless—no graphics or X11 required. Tested with OpenMPI 3.x/4.x, GCC 13, and SPARTA 2025-01-20.

---

## TL;DR Quickstart

```bash
# 1) Configure & build (from repo root)
cd ~/spaceforge-xai
rm -rf build && mkdir build && cd build
cmake -DSPARTA_DIR="$HOME/opt/sparta/src" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j

# 2) Run (dual, 2×2 ranks). Default cadence will print SPARTA stats frequently.
cd ~/spaceforge-xai/build
env -u DISPLAY mpirun -np 4 ./Sim/sim_app \
  --mode dual \
  --split 2 \
  --wake-deck in.wake \
  --eff-deck  in.effusion \
  --input-subdir input \
  --couple-every 10 \
  --sparta-block 200
```

### Faster smoke tests (no rebuild needed)
The app reads **all knobs at runtime**. Reduce work with either of these:

```bash
# ~500 SPARTA steps per instance total
env -u DISPLAY mpirun -np 4 ./Sim/sim_app \
  --mode dual --split 2 \
  --wake-deck in.wake --eff-deck in.effusion --input-subdir input \
  --couple-every 50 --sparta-block 50

# Super quick (~100 steps per instance)
env -u DISPLAY mpirun -np 4 ./Sim/sim_app \
  --mode dual --split 2 \
  --wake-deck in.wake --eff-deck in.effusion --input-subdir input \
  --couple-every 100 --sparta-block 10
```

> **Tip:** Your decks (`input/in.wake`, `input/in.effusion`) currently end with `run 1000`. Comment those out (or set `run 0`) to avoid a long startup burst and let the app fully drive the stepping.

---

## What the program does

- Initializes MPI and builds a small **power model**: `SolarArray → PowerBus → Battery`, plus a `HeaterBank` load.
- In **dual mode**, splits world ranks into two sub-communicators:
  - **Wake** (first `--split` ranks): reads **`in.wake`** and runs a persistent SPARTA instance.
  - **Effusion** (remaining ranks): reads **`in.effusion`** and runs another persistent SPARTA instance.
- Every **`--couple-every`** engine ticks, each SPARTA instance is advanced by **`--sparta-block`** steps **without re-reading** the input deck. This is logged by SPARTA as lines like “`Loop time ... for 50 steps`”.
- After the fixed engine loop completes, both SPARTA instances are shut down cleanly and MPI finalizes.

Default engine loop length is **500 ticks** at **dt = 0.1 s**, compiled into `main.cpp`. (You can change these defaults in code, see _Optional: add a `--ticks` flag_.)

---

## Directory layout

```
spaceforge-xai/
├─ CMakeLists.txt               # top-level build (sets Sim/sim_app)
├─ input/                       # SPARTA input area (cwd for decks)
│  ├─ in.wake                   # Wake simulation deck (Argon, inflow @ xlo)
│  ├─ in.effusion               # Effusion/MBE-cell deck (Argon emitter)
│  ├─ data/                     # species/models (e.g., ar.species, ar.vss)
│  └─ params.inc                # (optional) runtime parameters written by app
├─ Sim/
│  └─ CMakeLists.txt            # library + executable targets
├─ include/                     # public headers
│  ├─ Battery.hpp
│  ├─ EffusionCell.hpp          # (legacy/alt wrapper; new code uses WakeChamber)
│  ├─ HeaterBank.hpp
│  ├─ Logger.hpp
│  ├─ PowerBus.hpp
│  ├─ SimulationEngine.hpp
│  ├─ SolarArray.hpp
│  ├─ SpartaBridge.hpp
│  ├─ Subsystem.hpp
│  ├─ TickContext.hpp
│  ├─ TickPhaseEngine.hpp
│  └─ WakeChamber.hpp
└─ src/                         # implementations
   ├─ Battery.cpp
   ├─ EffusionCell.cpp          # (legacy/alt; safe to ignore in dual mode)
   ├─ HeaterBank.cpp
   ├─ Logger.cpp
   ├─ PowerBus.cpp
   ├─ SimulationEngine.cpp
   ├─ SolarArray.cpp
   ├─ SpartaBridge.cpp
   ├─ TickPhaseEngine.cpp
   ├─ WakeChamber.cpp
   └─ main.cpp
```

---

## Components (what each file does)

### `src/main.cpp`
- Command-line parser (no external deps).
- Sets up the power subsystems and the `SimulationEngine` (dt, initialize, tick, shutdown).
- **Dual mode:** splits ranks into **Wake** and **Effusion** subcommunicators and launches one `WakeChamber` per group (both are the same class; the name is historical).
- **Legacy mode:** single SPARTA instance on `MPI_COMM_WORLD`.
- Drives **periodic coupling** via `runIfDirtyOrAdvance(spartaBlock)` every `coupleEvery` ticks.
- Default engine ticks: **500** (`dt=0.1 s`).

### `include/SimulationEngine.hpp` + `src/SimulationEngine.cpp`
- Minimal engine to **tick multiple subsystems** at a uniform time step.
- API: `addSubsystem()`, `setTickStep()`, `initialize()`, `tick()`, `shutdown()`.

### `include/Subsystem.hpp`
- Interface for subsystems the engine can tick. Typically defines `initialize()`, `step(dt)`, `shutdown()`.

### `Power model` (toy example to have useful CPU-side work)
- **`PowerBus`**: tracks net generation/loads and voltage, provides attach points for sources/sinks.
- **`SolarArray`**: simple sunlight-to-power function, publishes power to `PowerBus`.
- **`Battery`**: integrates state-of-charge, supplies/absorbs power via `PowerBus`.
- **`HeaterBank`**: configurable load; in `main.cpp` we call `heater.setDemand(150.0)` as an example.

_All four derive from `Subsystem` and are added to the `SimulationEngine`._

### `include/SpartaBridge.hpp` + `src/SpartaBridge.cpp`
- Thin C++ wrapper over the **SPARTA C library interface**:
  - Opens the library with `sparta_open(argc, argv, comm, &spa_)`. We pass `-log log.capi` by default.
  - `runDeck(deck, subdir)`: `chdir` to `${PROJECT_SOURCE_DIR}/${subdir}` so **relative paths in decks** (e.g., `data/...`) resolve, then `sparta_file(deck)`.
  - `command("...")`: pass-through to `sparta_command` (used for `"clear"`, `"run N"`, etc.).
  - `runSteps(N)`, `clear()` helpers.
  - RAII close in destructor.

> `PROJECT_SOURCE_DIR` is provided via CMake (see below).

### `include/WakeChamber.hpp` + `src/WakeChamber.cpp`
- Higher-level façade that manages **one persistent SPARTA instance** on a specific `MPI_Comm`.
- `init(deck, inputDir)`: read once and keep alive.
- `runSteps(N)`: issue `"run N"` without re-reading.
- `markDirtyReload()` + `runIfDirtyOrAdvance(N)`: on next call it will `"clear"` and re-read the original deck, then optionally run.
- `setParameter(name, value)`: writes `input/params.inc` (rank 0 only); your decks can `include params.inc` to pick up runtime values.

### (Legacy / optional) `EffusionCell.hpp/cpp`
- Earlier single-instance helper; **not required** when using `WakeChamber` for both flows.

### Tick utilities
- `TickContext.hpp`, `TickPhaseEngine.hpp`: small helpers for staged per-tick execution (not heavily used in this scaffold).
- `Logger.hpp/cpp`: lightweight logging convenience.

---

## CMake notes

Minimum that matters here:
- Expose the SPARTA headers and library path: `-DSPARTA_DIR="$HOME/opt/sparta/src"`.
- Define `PROJECT_SOURCE_DIR` for `SpartaBridge::runDeck()` to resolve deck paths:

```cmake
# In Sim/CMakeLists.txt or the target where WakeChamber/SpartaBridge compile
target_compile_definitions(Simcore PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
```

If SPARTA shared objects aren’t on your default path at runtime, add:
```bash
export LD_LIBRARY_PATH="$HOME/opt/sparta/src:$LD_LIBRARY_PATH"
```

---

## Command-line options (app)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode {dual|legacy}` | `dual` | `dual`: split MPI ranks into two groups (Wake & Effusion). `legacy`: one SPARTA instance on `MPI_COMM_WORLD`. |
| `--split N` | `size/2` | In dual mode, number of ranks for **Wake**; the rest go to **Effusion**. |
| `--wake-deck FILE` | `in.wake` | Wake deck filename, resolved under `--input-subdir`. |
| `--eff-deck FILE` | `in.effusion` | Effusion deck filename, resolved under `--input-subdir`. |
| `--input-subdir DIR` | `input` | Directory (relative to project root) used as `cwd` before `sparta_file`. |
| `--couple-every T` | `10` | Every **T engine ticks**, advance SPARTA. |
| `--sparta-block N` | `200` | Number of SPARTA steps to run on each couple. |

> The engine currently runs a fixed **500 ticks** at **dt = 0.1 s** (see `main.cpp`).

### Optional: add a `--ticks` flag
If you want a runtime cap for the engine loop:
1. In `Args`, add `int ticks = 500;`  
2. In `parse_args()`, parse `--ticks`  
3. Replace `const int NTICKS = 500;` with `const int NTICKS = args.ticks;`  
Rebuild once and then you can do:
```bash
... ./Sim/sim_app --mode dual --split 2 --ticks 100 --couple-every 20 --sparta-block 25
```

---

## Decks (`input/`)

### `in.wake` (starter Argon wake)
- 3D, open boundaries, inflow from **xlo** using a drifting Maxwellian (`mixture air ...` with `Ar`).
- Sets `global nrho`, `global fnum` so **particle weight** is defined.
- Diagnostics: `stats`, `c_temp`.
- `timestep 7e-9` and `run 1000` (consider commenting out the `run 1000`).

### `in.effusion` (MBE-cell stand‑in)
- Small open box, grid 8×8×8.
- **Important fix:** include the species/mixture in the collide line. Either:

  **(A) collide the species directly)**
  ```sparta
  species data/ar.species Ar
  collide vss Ar data/ar.vss
  fix fe emit/face xlo species Ar temp 300.0 n 1.0e20 vstream 800.0 0.0 0.0
  ```
  **(B) or define a mixture and emit/collide it)**
  ```sparta
  species data/ar.species Ar
  mixture beam Ar temp 300.0 vstream 800.0 0 0
  collide vss beam data/ar.vss
  fix fe emit/face xlo beam
  ```

- To match wake particle weight, add the same **`global fnum`** used in `in.wake`:
  ```sparta
  global fnum 7.07043e6
  ```

- As with `in.wake`, consider removing the initial `run 1000` and let the app drive stepping.

> Both decks run with **open boundaries** and **no surfaces**. If/when you add geometry (shield, wafer), import a mesh and enable `surf_collide ...`; your current logs show all `Surface-* = 0`, which is expected for gas-only tests.

---

## Typical output & what it means

- Big block like “`Loop time ... for 1000 steps`” at start → comes from `run 1000` at the end of each deck.
- Later, smaller repeating blocks (“`for 50 steps`” or “`for 200 steps`”) → triggered by your **coupling cadence** (`--sparta-block` each time the engine reaches `--couple-every` ticks).
- It returns to the shell when **both** MPI sub-groups finish their 500-tick engine loops and `MPI_Finalize()` completes.

---

## Troubleshooting

- **`ERROR: Illegal collide command (collide_vss.cpp:45)`**  
  Supply a species/mixture name: `collide vss Ar data/ar.vss` (or create a `mixture` and use that ID).

- **“Authorization required, but no authorization protocol specified”**  
  Harmless X11 message; ignore—you're already running headless with `env -u DISPLAY`.

- **SPARTA library not found** at runtime  
  Add it to the loader path:  
  `export LD_LIBRARY_PATH="$HOME/opt/sparta/src:$LD_LIBRARY_PATH"`

- **Warning about ghost cells not clumped**  
  Benign for small tests. If you want to silence/optimize, try `balance_grid rcb clump part` or align grid decomposition with ranks.

- **Deck includes not found**  
  `SpartaBridge` changes CWD to `${PROJECT_SOURCE_DIR}/${input_subdir}` before `sparta_file`. Make sure CMake defines `PROJECT_SOURCE_DIR` for the target (see CMake notes).

- **Too much console spam**  
  Increase `stats` interval in the decks (e.g., `stats 1000`).

---

## License & acknowledgments

- SPARTA is © Sandia National Laboratories; see SPARTA’s own license for terms.
- This scaffold is for demonstration and research coupling patterns only.
- Contributions welcome via PRs (please keep the example decks small and self‑contained).
