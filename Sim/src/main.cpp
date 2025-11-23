// Sim/src/main.cpp
// mpirun -np 4 ./build/Sim/sim
/**
Build (from repo root):
  cd ~/spaceforge-xai
  rm -rf build && mkdir build && cd build
  cmake -DSPARTA_DIR="$HOME/opt/sparta/src" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . -j

Run (from build/, headless):
  env -u DISPLAY mpirun -np 4 ./Sim/sim                             // uses run.sh defaults
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode legacy               // original single-instance
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode wake                 // wake-only (no effusion ranks)
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode dual                 // alias of wake (no effusion)
  env -u DISPLAY mpirun -np 4 ./Sim/sim --mode wake \
    --wake-deck in.wake_harness --input-subdir input \
    --couple-every 10 --sparta-block 200

  // C++ harness only, no SPARTA; rank 0 logs to Sim/sim_debug_*.log
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
#include "orbit.hpp"   // simple circular orbit model
#include "Logger.hpp"  // for Orbit.csv logging

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
  double dt            = 60;         // seconds per engine tick now 60 from 0.1
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
  double heater_W    = 0.0;   // (legacy) heater demand in watts
};

// Map desired wafer flux to an approximate heater power demand.
// This is a placeholder calibration: tune F_low/F_high and P_low/P_high later.
static double fluxToHeaterPower(double Fwafer_cm2s) {
  // No beam then no heater
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
    return 0.0;
  }

  const double F_low  = 5.0e13;   // lower design flux
  const double F_high = 1.0e14;   // upper design flux

  const double P_low  = 120.0;    // heater power at F_low
  const double P_high = 180.0;    // heater power at F_high

  // Clamp flux into [F_low, F_high] for interpolation
  double F = Fwafer_cm2s;
  if (F < F_low)  F = F_low;
  if (F > F_high) F = F_high;

  double scale = (F - F_low) / (F_high - F_low);  // in [0,1]
  double P = P_low + scale * (P_high - P_low);

  // Safety clamp (HeaterBank max is 200 W in your ctor)
  if (P < 0.0)   P = 0.0;
  if (P > 200.0) P = 200.0;

  return P;
}

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
  // Sanity clamps so bad CLI/env values cannot kill the simulation loop.
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
        const std::string jobsPath = args.inputDir + "/jobs2.txt";
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
                << "] ticks " << j.start_tick << "-" << j.end_tick
                << ", Fwafer=" << j.Fwafer_cm2s
                << " cm^-2 s^-1, heater=" << j.heater_W << " W\n";
          }
          log_msg(oss.str());
        }
      }
    }

    // Broadcast number of jobs to all ranks (so everyone can make decisions
    // consistently if needed later).
    int njobs = static_cast<int>(jobs.size());
    MPI_Bcast(&njobs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --------------------------------------------------------------------
    // Lambda to write params.inc (leader only) and sync ranks.
    // Writes Fwafer_cm2s, mbe_active, and CUP_BASE_SCALE so SPARTA sees all.
    // Also clamps Fwafer_cm2s to a positive floor to avoid zero-density errors.
    // --------------------------------------------------------------------
    const double FWAFFER_FLOOR_CM2S = 1.0e8; // small but positive so mixture is legal

    auto write_params_inc = [&](double Fwafer_cm2s,
                                double mbe_active,
                                double cupola_scale) {
      // Clamp flux to positive floor
      if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
        Fwafer_cm2s = FWAFFER_FLOOR_CM2S;
      }
      if (!std::isfinite(mbe_active)) {
        mbe_active = 0.0;
      }
      if (!std::isfinite(cupola_scale)) {
        cupola_scale = 1.0; // neutral scale if something goes weird
      }

      if (rank == 0) {
        std::string path = args.inputDir + "/params.inc";
        std::ofstream out(path);
        if (!out) {
          std::ostringstream oss;
          oss << "[fatal] Cannot open " << path << " for writing.\n";
          log_msg(oss.str());
          throw std::runtime_error("Failed to write params.inc");
        }
        out << "variable Fwafer_cm2s  equal " << Fwafer_cm2s  << "\n";
        out << "variable mbe_active   equal " << mbe_active   << "\n";
        out << "variable CUP_BASE_SCALE equal " << cupola_scale << "\n";
        out.close();

        std::ostringstream oss;
        oss << "[params] Wrote params.inc: Fwafer_cm2s=" << Fwafer_cm2s
            << ", mbe_active=" << mbe_active
            << ", CUP_BASE_SCALE=" << cupola_scale << "\n";
        log_msg(oss.str());
      }

      // Make sure all ranks wait until file is written
      MPI_Barrier(MPI_COMM_WORLD);
    };

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

      // Seed params.inc BEFORE the first SPARTA deck load.
      double initial_Fwafer = FWAFFER_FLOOR_CM2S;
      if (rank == 0 && !jobs.empty()) {
        // Use first job's flux as a reasonable starting point
        initial_Fwafer = jobs.front().Fwafer_cm2s;
        if (!std::isfinite(initial_Fwafer) || initial_Fwafer <= 0.0) {
          initial_Fwafer = FWAFFER_FLOOR_CM2S;
        }
      }
      // Use neutral cupola scale = 1.0 for initial deck load.
      double initial_cupola_scale = 1.0;

      // Broadcast initial flux to all ranks so they agree
      MPI_Bcast(&initial_Fwafer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Beam off initially (mbe_active = 0), but flux positive so mixture is legal
      write_params_inc(initial_Fwafer, 0.0, initial_cupola_scale);

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

      // ---------------- Orbit model (leader drives logging + cupola scale) --
      // Simple circular LEO at 300 km altitude, time step = dt (engine tick).
      OrbitModel orbit(/*altitude_m=*/300e3,
                       /*dt_s=*/dt,
                       /*inclination_rad=*/0.0,
                       /*sun_theta_rad=*/0.0);

      if (isLeader) {
        std::ostringstream oss;
        oss << "[orbit] altitude_m=300000, period_s=" << orbit.period_s()
            << " (~" << orbit.period_s() / 60.0 << " min)\n";
        log_msg(oss.str());
      }

      // Track which job and parameters are currently active (leader only).
      const Job* currentJob = nullptr;
      int        currentJobIndex = -1;
      double     last_heater_set   = std::numeric_limits<double>::quiet_NaN();
      double     last_Fwafer_sent  = std::numeric_limits<double>::quiet_NaN();
      double     last_mbe_sent     = std::numeric_limits<double>::quiet_NaN();
      double     last_cupola_sent  = std::numeric_limits<double>::quiet_NaN();

      // Job health tracking (leader only)
      std::vector<bool> jobAborted(jobs.size(), false);
      int   underflux_streak          = 0;
      const int    UNDERFLUX_LIMIT_TICKS = 5;   // consecutive ticks
      const double MIN_FLUX_FRACTION     = 0.99;  // require >= 99% of requested power

      const int NTICKS = args.nticks;
      constexpr double PI_MAIN = 3.141592653589793;

      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;

        // ---------------- leader: orbit, job schedule, per-tick harness -----
        double cupola_scale_this_tick = 1.0;

        if (isLeader) {
          // ---- 0) Orbit update + logging ----
          orbit.step();
          const OrbitState &orb = orbit.state();

          double t_min     = orb.t_orbit_s / 60.0;
          double theta_deg = orb.theta_rad * (180.0 / PI_MAIN);
          cupola_scale_this_tick = orb.solar_scale; // 0 in eclipse, 1 in sun for now

          // Log via the same Logger as other subsystems: creates Orbit.csv
          Logger::instance().log_wide(
              "Orbit",
              tickIndex,
              t_phys,
              {"t_orbit_s","t_orbit_min","theta_rad","theta_deg","in_sun","solar_scale"},
              {orb.t_orbit_s, t_min, orb.theta_rad, theta_deg,
               orb.in_sun ? 1.0 : 0.0, orb.solar_scale}
          );

          // ---- 1) Determine active job for this tick (if any) ----
          const Job* newJob = nullptr;
          int        newJobIndex = -1;

          if (!jobs.empty()) {
            for (std::size_t idx = 0; idx < jobs.size(); ++idx) {
              if (jobAborted[idx]) continue;  // skip jobs we have already killed
              const Job& j = jobs[idx];
              if (tickIndex >= j.start_tick && tickIndex <= j.end_tick) {
                newJob      = &j;
                newJobIndex = static_cast<int>(idx);
                break;
              }
            }
          }

          if (newJob != currentJob) {
            // Reset underflux streak when entering/leaving a job
            underflux_streak = 0;

            std::ostringstream oss;
            if (newJob) {
              oss << "[job] tick=" << tickIndex
                  << " entering job window ["
                  << newJob->start_tick << ","
                  << newJob->end_tick << "] (index=" << newJobIndex << ") "
                  << "Fwafer_cm2s=" << newJob->Fwafer_cm2s
                  << ", heater_W=" << newJob->heater_W << "\n";
            } else if (currentJob) {
              oss << "[job] tick=" << tickIndex
                  << " leaving job window; reverting to baseline (heater=0, beam off).\n";
            }
            log_msg(oss.str());

            currentJob      = newJob;
            currentJobIndex = newJobIndex;
          }

          // ---- 2) Decide heater demand, Fwafer, and mbe_active for this tick ----
          double heaterDemand_W = 150.0;   // baseline if no jobs.txt
          double Fwafer_cmd     = std::numeric_limits<double>::quiet_NaN();
          double mbe_flag       = 0.0;     // 0 = beam off, 1 = beam on

          if (!jobs.empty()) {
            if (currentJob) {
              Fwafer_cmd     = currentJob->Fwafer_cm2s;
              heaterDemand_W = fluxToHeaterPower(Fwafer_cmd);
              mbe_flag       = 1.0;
            } else {
              // Outside any job window or after abort: effusion off, heater 0
              heaterDemand_W = 0.0;
              // We do NOT set Fwafer_cmd to 0.0; that would kill mixture.
              // Instead, keep last flux (or floor) and just set mbe_active = 0.
              if (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0) {
                Fwafer_cmd = FWAFFER_FLOOR_CM2S;
              } else {
                Fwafer_cmd = last_Fwafer_sent;
              }
              mbe_flag = 0.0;
            }
          } else {
            // No jobs.txt: keep baseline heater and floor flux, beam off
            heaterDemand_W = 150.0;
            if (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0) {
              Fwafer_cmd = FWAFFER_FLOOR_CM2S;
            } else {
              Fwafer_cmd = last_Fwafer_sent;
            }
            mbe_flag = 0.0;
          }

          // ---- 3) Push Fwafer + mbe_active + CUP_BASE_SCALE into params.inc
          //         when any of them change.
          if (!std::isnan(Fwafer_cmd)) {
            bool need_update = false;

            if (std::isnan(last_Fwafer_sent) ||
                Fwafer_cmd != last_Fwafer_sent) {
              need_update = true;
            }
            if (std::isnan(last_mbe_sent) ||
                mbe_flag != last_mbe_sent) {
              need_update = true;
            }
            if (std::isnan(last_cupola_sent) ||
                cupola_scale_this_tick != last_cupola_sent) {
              need_update = true;
            }

            if (need_update) {
              std::ostringstream oss;
              oss << "[job] tick=" << tickIndex
                  << " update params.inc: Fwafer_cm2s=" << Fwafer_cmd
                  << ", mbe_active=" << mbe_flag
                  << ", CUP_BASE_SCALE=" << cupola_scale_this_tick << "\n";
              log_msg(oss.str());

              write_params_inc(Fwafer_cmd, mbe_flag, cupola_scale_this_tick);
              wake.markDirtyReload();

              last_Fwafer_sent = Fwafer_cmd;
              last_mbe_sent    = mbe_flag;
              last_cupola_sent = cupola_scale_this_tick;
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

          // C++ harness first, then SPARTA
          engine.tick();

          TickContext ctx{ tickIndex, t_phys, dt };
          wake.tick(ctx); 

          std::ostringstream oss2;
          oss2 << "[wake] tick=" << tickIndex
               << " t=" << t_phys
               << " s : AFTER engine.tick() + wake.tick()\n";
          log_msg(oss2.str());

          // ---- 6) After ticking, evaluate job health (under-flux guard) ----
          if (currentJob && heaterDemand_W > 1e-6) {
            double P_actual = effCell.getLastHeatInputW();
            double flux_ratio = 1.0;

            if (heaterDemand_W > 0.0) {
              flux_ratio = P_actual / heaterDemand_W;
            }
            if (!std::isfinite(flux_ratio)) {
              flux_ratio = 0.0;
            }

            if (flux_ratio < MIN_FLUX_FRACTION) {
              underflux_streak += 1;
            } else {
              underflux_streak = 0;
            }

            if (underflux_streak >= UNDERFLUX_LIMIT_TICKS &&
                currentJobIndex >= 0 &&
                currentJobIndex < static_cast<int>(jobAborted.size()) &&
                !jobAborted[currentJobIndex]) {

              jobAborted[currentJobIndex] = true;

              std::ostringstream joss;
              joss << "[job] tick=" << tickIndex
                   << " ABORTING job index " << currentJobIndex
                   << " due to under-flux for " << underflux_streak
                   << " consecutive ticks (flux_ratio=" << flux_ratio << ")\n";
              log_msg(joss.str());

              // Mark failure for SimulationEngine.csv on this tick
              engine.markJobFailedThisTick();

              // Immediately tell SPARTA the beam is off.
              double F_for_abort = last_Fwafer_sent;
              if (!std::isfinite(F_for_abort) || F_for_abort <= 0.0) {
                F_for_abort = FWAFFER_FLOOR_CM2S;
              }
              write_params_inc(F_for_abort, 0.0, cupola_scale_this_tick);
              wake.markDirtyReload();
              last_Fwafer_sent = F_for_abort;
              last_mbe_sent    = 0.0;
              last_cupola_sent = cupola_scale_this_tick;

              // Reset job state so next tick we fall into "no job" path
              currentJob       = nullptr;
              currentJobIndex  = -1;
              underflux_streak = 0;
              last_heater_set  = std::numeric_limits<double>::quiet_NaN();
            }
          }
        } // end if (isLeader)

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
