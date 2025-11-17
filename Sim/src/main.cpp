// Sim/src/main.cpp
// mpirun -np 4 ./build/Sim/sim
/**
Build (from repo root):
  cd ~/spaceforge-xai
  rm -rf build && mkdir build && cd build
  cmake -DSPARTA_DIR="$HOME/opt/sparta/src" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . -j

Run (from build/, headless):
  env -u DISPLAY mpirun -np 4 ./Sim/sim                             # uses run.sh defaults
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode legacy               # original single-instance
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode wake                 # wake-only (no effusion ranks)
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode dual                 # alias of wake (no effusion)
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode wake \
    --wake-deck in.wake_harness --input-subdir input \
    --couple-every 10 --sparta-block 200

  # C++ harness only, no SPARTA; rank 0 logs to Sim/sim_debug_*.log
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode power --nticks 500 --dt 0.1
*/

#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>

#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"
#include "WakeChamber.hpp"
#include "EffusionCell.hpp"

// ---------------------------
// Tiny CLI helpers (no deps)
// ---------------------------
struct Args {
  std::string mode     = "dual";     // "dual", "legacy", "wake", or "power"
  std::string wakeDeck = "in.wake_harness";
  std::string effDeck  = "in.effusion"; // kept for compatibility, but unused now
  std::string inputDir = "input";
  int  nWake           = -1;         // unused now (no dual effusion), kept for compat
  int  coupleEvery     = 10;         // advance SPARTA every X engine ticks
  int  spartaBlock     = 200;        // run N steps per advance
  bool showHelp        = false;

  // make tick size and run length configurable
  int    nticks        = 500;        // engine ticks to run
  double dt            = 0.1;        // seconds per engine tick
};

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a,b) == 0; }

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--mode") && i+1 < argc)              a.mode = argv[++i];
    else if (arg_eq(argv[i], "--wake-deck") && i+1 < argc)    a.wakeDeck = argv[++i];
    else if (arg_eq(argv[i], "--eff-deck") && i+1 < argc)     a.effDeck  = argv[++i];  // ignored now
    else if (arg_eq(argv[i], "--input-subdir") && i+1 < argc) a.inputDir = argv[++i];
    else if (arg_eq(argv[i], "--split") && i+1 < argc)        a.nWake = std::atoi(argv[++i]); // ignored
    else if (arg_eq(argv[i], "--couple-every") && i+1 < argc) a.coupleEvery = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--sparta-block") && i+1 < argc) a.spartaBlock = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--nticks") && i+1 < argc)       a.nticks = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--dt") && i+1 < argc)           a.dt = std::atof(argv[++i]);
    else if (arg_eq(argv[i], "--help"))                       a.showHelp = true;
  }
  return a;
}

static void print_usage() {
  std::cout <<
    "Usage: sim [--mode dual|legacy|wake|power]\n"
    "           [--wake-deck in.wake_harness]\n"
    "           [--input-subdir input]\n"
    "           [--couple-every T] [--sparta-block N]\n"
    "           [--nticks N] [--dt seconds]\n"
    "\n"
    "Modes:\n"
    "  legacy  - single SPARTA instance on MPI_COMM_WORLD (wake only)\n"
    "  wake    - wake-only with in.wake_harness, no effusion ranks\n"
    "  dual    - alias of wake (same as wake, no separate effusion deck)\n"
    "  power   - C++ power/thermal harness only (no SPARTA)\n"
    "\n"
    "Default 'dual' is currently an alias of 'wake'; both run a single wake\n"
    "deck across MPI_COMM_WORLD. Coupling advances SPARTA by N steps every\n"
    "T engine ticks.\n";
}

// ---------------------------
// Job schedule for effusion
// ---------------------------
struct Job {
  int    start_tick = 0;      // inclusive
  int    end_tick   = -1;     // inclusive
  double Fwafer_cm2s = 0.0;   // effusion flux to send to SPARTA
  double heater_W    = 0.0;   // heater demand in watts
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Args args = parse_args(argc, argv);

  // ------------------------------------------------------------------------
  // Debug logger: mirrors messages to stderr and a per-run file on rank 0.
  // Log file name: sim_debug_<RUN_ID>_<mode>.log (in Sim/).
  // ------------------------------------------------------------------------
  std::ofstream debugLog;
  auto log_msg = [&](const std::string& s) {
    if (rank == 0) {
      std::cerr << s;
      if (debugLog.is_open()) {
        debugLog << s;
        debugLog.flush();
      }
    }
  };

  // Open log file on rank 0 after we know the mode/env.
  if (rank == 0) {
    const char* env_run_id = std::getenv("RUN_ID");
    std::string run_id     = env_run_id ? env_run_id : "norunid";
    std::string mode_tag   = args.mode.empty() ? "nomode" : args.mode;

    std::string filename = "sim_debug_" + run_id + "_" + mode_tag + ".log";
    debugLog.open(filename, std::ios::out | std::ios::app);
    if (!debugLog) {
      std::cerr << "[warn] Failed to open " << filename << " for writing.\n";
    } else {
      debugLog << "============================================================\n";
      debugLog << "New run started (mode=" << args.mode
               << ", RUN_ID=" << run_id << ", world_size=" << size << ")\n";
      debugLog << "============================================================\n";
      debugLog.flush();
    }
  }

  if (args.showHelp) {
    if (rank == 0) print_usage();
    MPI_Finalize();
    return 0;
  }

  // ------------------------------------------------------------------------
  // Sanity clamps so bad CLI/env values can't kill the simulation loop.
  // ------------------------------------------------------------------------
  if (args.nticks <= 0) {
    if (rank == 0) log_msg("[warn] nticks <= 0 from CLI/env; defaulting to 500.\n");
    args.nticks = 500;
  }

  if (args.dt <= 0.0) {
    if (rank == 0) log_msg("[warn] dt <= 0 from CLI/env; defaulting to 0.1 s.\n");
    args.dt = 0.1;
  }

  if (args.coupleEvery <= 0) {
    if (rank == 0) log_msg("[warn] couple-every <= 0; defaulting to 10.\n");
    args.coupleEvery = 10;
  }

  if (args.spartaBlock <= 0) {
    if (rank == 0) log_msg("[warn] sparta-block <= 0; defaulting to 200.\n");
    args.spartaBlock = 200;
  }

  try {
    // --------------------------------------------------------------------
    // Dump CLI args and key env vars to help with debugging.
    // --------------------------------------------------------------------
    if (rank == 0) {
      std::ostringstream oss;
      oss << "[info] MPI world size = " << size << "\n";
      oss << "[info] Args: mode=" << args.mode
          << " wakeDeck=" << args.wakeDeck
          << " effDeck=" << args.effDeck
          << " inputDir=" << args.inputDir
          << " nWake=" << args.nWake
          << " nticks=" << args.nticks
          << " dt=" << args.dt
          << " coupleEvery=" << args.coupleEvery
          << " spartaBlock=" << args.spartaBlock << "\n";

      const char* env_run_id        = std::getenv("RUN_ID");
      const char* env_enable_sparta = std::getenv("ENABLE_SPARTA");
      const char* env_mode          = std::getenv("MODE");
      const char* env_input_subdir  = std::getenv("INPUT_SUBDIR");

      oss << "[info] Env: RUN_ID="        << (env_run_id        ? env_run_id        : "<unset>") << "\n";
      oss << "[info] Env: ENABLE_SPARTA=" << (env_enable_sparta ? env_enable_sparta : "<unset>") << "\n";
      oss << "[info] Env: MODE="          << (env_mode          ? env_mode          : "<unset>") << "\n";
      oss << "[info] Env: INPUT_SUBDIR="  << (env_input_subdir  ? env_input_subdir  : "<unset>") << "\n";

      log_msg(oss.str());
    }

    // --------------------------------------------------------------------
    // Load jobs.txt (only rank 0 actually uses it; others just follow MPI)
    // --------------------------------------------------------------------
    std::vector<Job> jobs;
    if (args.mode == "wake" || args.mode == "dual" || args.mode == "legacy") {
      if (rank == 0) {
        const std::string jobsPath = args.inputDir + "/jobs.txt";
        std::ifstream jf(jobsPath);
        if (!jf) {
          std::ostringstream oss;
          oss << "[info] No jobs.txt found at " << jobsPath
              << " — running with default heater/flux.\n";
          log_msg(oss.str());
        } else {
          std::string line;
          int lineno = 0;
          while (std::getline(jf, line)) {
            ++lineno;
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            std::istringstream iss(line);
            Job j{};
            if (!(iss >> j.start_tick >> j.end_tick >> j.Fwafer_cm2s >> j.heater_W)) {
              std::ostringstream oss;
              oss << "[warn] jobs.txt line " << lineno
                  << " malformed, skipping: " << line << "\n";
              log_msg(oss.str());
              continue;
            }
            if (j.end_tick < j.start_tick) std::swap(j.start_tick, j.end_tick);
            jobs.push_back(j);
          }

          std::ostringstream oss;
          oss << "[info] Loaded " << jobs.size() << " job(s) from " << jobsPath << "\n";
          for (std::size_t i = 0; i < jobs.size(); ++i) {
            const Job& j = jobs[i];
            oss << "  [job " << i
                << "] ticks " << j.start_tick << "–" << j.end_tick
                << ", Fwafer=" << j.Fwafer_cm2s
                << " cm^-2 s^-1, heater=" << j.heater_W << " W\n";
          }
          log_msg(oss.str());
        }
      }
    }

    // --------------------------------------------------------------------
    // Electrical/power subsystems (independent of SPARTA)
    // --------------------------------------------------------------------
    PowerBus   bus;
    SolarArray solar;
    Battery    battery;
    HeaterBank heater(/*maxPowerW=*/200.0);
    EffusionCell effCell;

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);
    heater.setEffusionCell(&effCell);     // Heater warms the effusion cell

    SimulationEngine engine;
    engine.addSubsystem(&effCell);
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    const double dt = args.dt;
    engine.setTickStep(dt);
    engine.initialize();

    if (rank == 0) {
      std::ostringstream oss;
      oss << "[info] Simulation starting on " << size << " MPI task(s)\n";
      oss << "[info] Mode = "          << args.mode        << "\n";
      oss << "[info] nticks = "        << args.nticks      << "\n";
      oss << "[info] dt = "            << dt               << " s\n";
      oss << "[info] couple-every = "  << args.coupleEvery << "\n";
      oss << "[info] sparta-block = "  << args.spartaBlock << "\n";
      log_msg(oss.str());
    }

    // ======================================================================
    // MODE: power (C++ harness only, no SPARTA / no WakeChamber)
    // ======================================================================
    if (args.mode == "power") {
      if (rank == 0) {
        log_msg("[info] Entering power-only mode (no SPARTA / no WakeChamber).\n");
      }

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;

        if (rank == 0) {
          std::ostringstream oss;
          oss << "[power] tick=" << tickIndex
              << " t=" << t_phys << " s : calling engine.tick()\n";
          log_msg(oss.str());

          heater.setDemand(150.0);
          engine.tick();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (rank == 0) {
        log_msg("[info] power-only loop completed; shutting down engine.\n");
      }

      engine.shutdown();
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      return EXIT_SUCCESS;
    }

    // ======================================================================
    // MODE: legacy / wake / dual (all are wake-only now, harness-driven)
    // ======================================================================
    if (args.mode == "legacy" || args.mode == "wake" || args.mode == "dual") {
      if (rank == 0 && args.mode == "dual") {
        log_msg("[info] dual mode selected; using wake-only path (no effusion deck).\n");
      }

      if (rank == 0) {
        log_msg("[info] Constructing WakeChamber and calling wake.init(...)\n");
      }
      WakeChamber wake(MPI_COMM_WORLD, "WakeChamber");
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());
      if (rank == 0) {
        log_msg("[info] wake.init() returned; entering main wake loop.\n");
      }

      int worldrank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
      const bool isLeader = (worldrank == 0);

      // Track which job and parameters are currently active (leader only).
      const Job* currentJob = nullptr;
      double last_heater_set   = std::numeric_limits<double>::quiet_NaN();
      double last_Fwafer_sent  = std::numeric_limits<double>::quiet_NaN();

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;

        // ---------------- leader: job schedule + per-tick harness -----------
        if (isLeader) {
          // ---- 1) Determine active job for this tick (if any) ----
          const Job* newJob = nullptr;
          if (!jobs.empty()) {
            for (const Job& j : jobs) {
              if (tickIndex >= j.start_tick && tickIndex <= j.end_tick) {
                newJob = &j;
                break;
              }
            }
          }

          if (newJob != currentJob) {
            std::ostringstream oss;
            if (newJob) {
              oss << "[job] tick=" << tickIndex
                  << " entering job window ["
                  << newJob->start_tick << ","
                  << newJob->end_tick << "] "
                  << "Fwafer_cm2s=" << newJob->Fwafer_cm2s
                  << ", heater_W=" << newJob->heater_W << "\n";
            } else if (currentJob) {
              oss << "[job] tick=" << tickIndex
                  << " leaving job window; reverting to baseline (heater=0, Fwafer=0).\n";
            }
            log_msg(oss.str());
            currentJob = newJob;
          }

          // ---- 2) Decide heater demand and Fwafer for this tick ----
          double heaterDemand_W = 150.0;   // baseline if no jobs.txt
          double Fwafer_cmd     = std::numeric_limits<double>::quiet_NaN();

          if (!jobs.empty()) {
            if (currentJob) {
              heaterDemand_W = currentJob->heater_W;
              Fwafer_cmd     = currentJob->Fwafer_cm2s;
            } else {
              // Outside any job window: effusion off, heater 0
              heaterDemand_W = 0.0;
              Fwafer_cmd     = 0.0;
            }
          }

          // ---- 3) Push Fwafer into SPARTA when it changes ----
          if (!std::isnan(Fwafer_cmd)) {
            if (std::isnan(last_Fwafer_sent) ||
                Fwafer_cmd != last_Fwafer_sent) {
              std::ostringstream oss;
              oss << "[job] tick=" << tickIndex
                  << " set SPARTA param Fwafer_cm2s=" << Fwafer_cmd << "\n";
              log_msg(oss.str());

              wake.setParameter("Fwafer_cm2s", Fwafer_cmd);
              wake.markDirtyReload();
              last_Fwafer_sent = Fwafer_cmd;
            }
          }

          // ---- 4) Set heater demand (only log when it changes) ----
          if (std::isnan(last_heater_set) ||
              heaterDemand_W != last_heater_set) {
            std::ostringstream oss;
            oss << "[job] tick=" << tickIndex
                << " set heater demand=" << heaterDemand_W << " W\n";
            log_msg(oss.str());
            last_heater_set = heaterDemand_W;
          }

          heater.setDemand(heaterDemand_W);

          // ---- 5) Tick harness + WakeChamber ----
          std::ostringstream oss;
          oss << "[wake] tick=" << tickIndex
              << " t=" << t_phys
              << " s : BEFORE engine.tick() + wake.tick()\n";
          log_msg(oss.str());

          engine.tick();

          TickContext ctx{ tickIndex, t_phys, dt };
          wake.tick(ctx);

          std::ostringstream oss2;
          oss2 << "[wake] tick=" << tickIndex
               << " t=" << t_phys
               << " s : AFTER engine.tick() + wake.tick()\n";
          log_msg(oss2.str());
        }

        // ---------------- SPARTA coupling block ----------------
        if (i % args.coupleEvery == 0) {
          if (isLeader) {
            std::ostringstream oss;
            oss << "[cpl] tick=" << (i + 1)
                << " ENTER wake.runIfDirtyOrAdvance(spartaBlock="
                << args.spartaBlock << ")\n";
            log_msg(oss.str());
          }

          wake.runIfDirtyOrAdvance(args.spartaBlock);

          if (isLeader) {
            std::ostringstream oss;
            oss << "[cpl] tick=" << (i + 1)
                << " EXIT  wake.runIfDirtyOrAdvance(...)\n";
            log_msg(oss.str());
          }
        }

        // Ensure all ranks stay roughly in sync
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (rank == 0) {
        log_msg("[info] wake main loop completed; shutting down.\n");
      }

      wake.shutdown();
      engine.shutdown();
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      return EXIT_SUCCESS;
    }

    // ----------------------------------------------------------------------
    // Unknown mode
    // ----------------------------------------------------------------------
    if (rank == 0) {
      std::ostringstream oss;
      oss << "[fatal] Unknown mode '" << args.mode
          << "'. Expected 'dual', 'legacy', 'wake', or 'power'.\n";
      log_msg(oss.str());
      print_usage();
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return EXIT_FAILURE;
  }
  catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "[fatal] std::exception on rank " << rank << ": " << e.what() << "\n";
    log_msg(oss.str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return EXIT_FAILURE;
  }
  catch (...) {
    std::ostringstream oss;
    oss << "[fatal] Unknown non-std exception on rank " << rank << "\n";
    log_msg(oss.str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return EXIT_FAILURE;
  }
}
