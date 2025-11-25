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
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>  // for std::clamp

#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"
#include "WakeChamber.hpp"
#include "EffusionCell.hpp"
#include "orbit.hpp"         // simple circular orbit model
#include "Logger.hpp"        // for Orbit.csv logging
#include "GrowthMonitor.hpp" // wafer dose / heatmap tracker
#include "helpers.hpp"       // new helpers split from main

// Bring helper types/functions into local scope
using SimHelpers::Args;
using SimHelpers::Job;
using SimHelpers::parse_args;
using SimHelpers::print_usage;
using SimHelpers::fluxToHeaterPower;
using SimHelpers::write_params_inc;
using SimHelpers::FWAFFER_FLOOR_CM2S;
using SimHelpers::LogFn;

// Globals defined in EffusionCell.cpp so streaks show up in EffusionCell.csv.
extern int g_underflux_streak_for_log;
extern int g_temp_miss_streak_for_log;

// Global sunlight scale shared with SolarArray.cpp.
// In wake/dual/legacy mode this is updated each tick from OrbitModel.
// In power mode (no orbit), it stays at 1.0 (always sunlit).
double g_orbit_solar_scale = 1.0;

// ---------------------------------------------------------------------------
// Map wafer flux (cm^-2 s^-1) from jobs.txt to a notional effusion-cell
// target temperature (K). This does NOT enforce the temperature; it simply
// provides a "desired" setpoint that we log as target_temp_K in EffusionCell.
// This is the original 1100–1300 K mapping you already had.
// ---------------------------------------------------------------------------
static double targetTempForFlux(double Fwafer_cm2s) {
  // Idle / no-flux baseline
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
    return 300.0;
  }

  // Design anchors: tweak as desired.
  const double F_low  = 5e13;   // lower design flux (cm^-2 s^-1)
  const double F_high = 1e14;   // upper design flux (cm^-2 s^-1)
  const double T_low  = 1100.0; // effusion temp at F_low (K)
  const double T_high = 1300.0; // effusion temp at F_high (K)

  // Clamp flux to the design band.
  double F_clamped = std::clamp(Fwafer_cm2s, F_low, F_high);

  double logF     = std::log(F_clamped);
  double logFlow  = std::log(F_low);
  double logFhigh = std::log(F_high);
  double denom    = (logFhigh - logFlow);

  double alpha = 0.0;
  if (denom > 0.0) {
    alpha = (logF - logFlow) / denom;
  }
  alpha = std::clamp(alpha, 0.0, 1.0);

  return T_low + alpha * (T_high - T_low);
}

// ---------------------------------------------------------------------------
// Estimate how many ticks of "warm-up" to ignore gate penalties for a job,
// based on its flux (via heater power) and the same RC model used for
// temp_proxy_K and EffusionCell::applyHeat. This does NOT change flux or
// SPARTA runs; it only controls when gates become active.
// ---------------------------------------------------------------------------
static int estimateWarmupTicksForFlux(double Fwafer_cm2s, double dt_s) {
  // Same RC constants as in the temp_proxy_K update below.
  const double C_J_PER_K = 1000.0;
  const double H_W_PER_K = 1.5;
  const double T_ENV_K   = 300.0;

  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    return 0;
  }

  // Map flux -> heater power using the helper.
  double P_W = fluxToHeaterPower(Fwafer_cm2s);
  if (!std::isfinite(P_W) || P_W <= 0.0) {
    return 0;
  }

  // Use the same target temperature mapping as logging.
  double T_target_K = targetTempForFlux(Fwafer_cm2s);
  if (!std::isfinite(T_target_K) || T_target_K <= T_ENV_K + 10.0) {
    return 0;
  }

  // Steady-state temperature under constant P_W.
  double T_ss_K = T_ENV_K + P_W / H_W_PER_K;
  if (T_ss_K <= T_ENV_K + 1.0) {
    return 0;
  }

  // We arm the gate once we reach some fraction of the target temperature.
  const double GATE_FRACTION = 0.9; // 90% of target
  double T_gate_K = GATE_FRACTION * T_target_K;

  // If target is above steady-state, fall back to fraction of T_ss.
  if (T_gate_K >= T_ss_K) {
    T_gate_K = 0.9 * T_ss_K;
  }

  double numer = T_gate_K - T_ENV_K;
  double denom = T_ss_K   - T_ENV_K;
  if (numer <= 0.0 || denom <= 0.0) {
    return 0;
  }

  double ratio = numer / denom;
  if (ratio >= 1.0) ratio = 0.999;
  if (ratio <= 0.0) ratio = 0.0;

  // First-order RC time constant.
  double tau_s = C_J_PER_K / H_W_PER_K;
  double t_gate_s = -tau_s * std::log(1.0 - ratio);

  if (!std::isfinite(t_gate_s) || t_gate_s <= 0.0) {
    return 0;
  }

  int ticks = static_cast<int>(std::ceil(t_gate_s / dt_s));
  if (ticks < 0) ticks = 0;

  // Safety cap so we never "warm up" longer than a reasonable window.
  const int MAX_WARMUP_TICKS = 60; // config: 60 engine ticks max
  if (ticks > MAX_WARMUP_TICKS) {
    ticks = MAX_WARMUP_TICKS;
  }

  return ticks;
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
  LogFn log_msg = [&](const std::string& s) {
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
            if (j.end_tick < j.start_tick) std::swap(j.end_tick, j.start_tick);
            jobs.push_back(j);
          }

          std::ostringstream oss;
          oss << "[info] Loaded " << jobs.size() << " job(s) from " << jobsPath << "\n";
          log_msg(oss.str());
          for (std::size_t i = 0; i < jobs.size(); ++i) {
            const Job& j = jobs[i];
            std::ostringstream joss;
            joss << "  [job " << i
                 << "] ticks " << j.start_tick << "-" << j.end_tick
                 << ", Fwafer=" << j.Fwafer_cm2s
                 << " cm^-2 s^-1, heater=" << j.heater_W << " W\n";
            log_msg(joss.str());
          }
        }
      }
    }

    // Broadcast number of jobs to all ranks (so everyone can make decisions
    // consistently if needed later).
    int njobs = static_cast<int>(jobs.size());
    MPI_Bcast(&njobs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Per-job dynamic warm-up ticks (leader only actually uses values).
    std::vector<int> jobWarmupTicks;
    if (rank == 0 && njobs > 0 &&
        (args.mode == "wake" || args.mode == "dual" || args.mode == "legacy")) {
      jobWarmupTicks.resize(jobs.size(), 0);
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const Job& j = jobs[i];
        int W = estimateWarmupTicksForFlux(j.Fwafer_cm2s, args.dt);
        jobWarmupTicks[i] = W;

        std::ostringstream oss;
        oss << "[info] Job " << i
            << " dynamic warm-up estimate: " << W
            << " tick(s) at dt=" << args.dt
            << " s (Fwafer_cm2s=" << j.Fwafer_cm2s << ")\n";
        log_msg(oss.str());
      }
    }

    // --------------------------------------------------------------------
    // Electrical/power subsystems (independent of SPARTA)
    // --------------------------------------------------------------------
    PowerBus      bus;
    SolarArray    solar;
    Battery       battery;
    bus.setBattery(&battery);
    // Bigger heater: can draw up to 2 kW from the bus.
    HeaterBank    heater(/*maxDraw=*/2000.0);
    EffusionCell  effCell;
    GrowthMonitor growth(/*gridN=*/32);

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);
    heater.setEffusionCell(&effCell);     // Heater warms the effusion cell
    growth.setPowerBus(&bus);             // Film/growth monitor draws instrument power

    // GrowthMonitor should only log + write CSV on leader, but engine tick
    // will be called on all ranks.
    growth.setIsLeader(rank == 0);
    growth.setNumJobs(jobs.size());

    SimulationEngine engine;
    engine.addSubsystem(&solar);      // 1) power source
    engine.addSubsystem(&battery);    // 2) storage update
    engine.addSubsystem(&heater);     // 3) power load
    engine.addSubsystem(&effCell);    // 4) heat response (after heater!)
    engine.addSubsystem(&bus);        // 5) bookkeeping on power totals
    engine.addSubsystem(&growth);     // 6) sensors/aux

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

      // Always treat as sunlit in power-only tests.
      g_orbit_solar_scale = 1.0;

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;

        if (rank == 0) {
          std::ostringstream oss;
          oss << "[power] tick=" << tickIndex
              << " t=" << t_phys << " s : calling engine.tick()\n";
          log_msg(oss.str());

          heater.setDemand(1500.0); // representative kW-scale test
          // No jobs in power-only mode -> growth monitor gets jobIndex=-1, mbeOff.
          growth.setBeamState(-1, false, 0.0);
          engine.tick();
        } else {
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

      // Broadcast initial flux to all ranks so they agree
      MPI_Bcast(&initial_Fwafer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Beam off initially (mbe_active = 0), but flux positive so mixture is legal
      write_params_inc(initial_Fwafer, 0.0, rank, args.inputDir, log_msg);

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

      // ---------------- Orbit model (leader drives logging + SolarArray) ---
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

      // Job health tracking (leader only)
      std::vector<bool> jobAborted(jobs.size(), false);
      int   underflux_streak            = 0;
      int   temp_miss_streak            = 0;
      int   job_tick_counter            = 0;
      const int    UNDERFLUX_LIMIT_TICKS    = 5;     // consecutive ticks
      const double MIN_FLUX_FRACTION        = 0.99;  // require >= 99% of requested power
      const int    TEMP_FAIL_LIMIT_TICKS    = 5;     // N consecutive temp misses
      const double TEMP_TOLERANCE_FRACTION  = 0.95;  // must reach >= 95% of target

      // RC temp proxy (mirrors EffusionCell RC constants) used for temp gate.
      double temp_proxy_K = 300.0;

      const int NTICKS = args.nticks;
      constexpr double PI_MAIN = 3.141592653589793;

      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;

        // ---------------- leader: orbit, job schedule, per-tick harness -----
        if (isLeader) {
          // ---- 0) Orbit update + logging ----
          orbit.step();
          const OrbitState &orb = orbit.state();

          double t_min     = orb.t_orbit_s / 60.0;
          double theta_deg = orb.theta_rad * (180.0 / PI_MAIN);

          // Update global sunlight scale for SolarArray.
          g_orbit_solar_scale = orb.solar_scale;

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
            // Reset streaks and per-job counters when entering/leaving a job
            underflux_streak   = 0;
            temp_miss_streak   = 0;
            job_tick_counter   = 0;
            g_underflux_streak_for_log = 0;
            g_temp_miss_streak_for_log = 0;
            temp_proxy_K       = 300.0;

            std::ostringstream oss;
            if (newJob) {
              oss << "[job] tick=" << tickIndex
                  << " entering job window ["
                  << newJob->start_tick << "," << newJob->end_tick
                  << "] (index=" << newJobIndex << ") "
                  << "Fwafer_cm2s=" << newJob->Fwafer_cm2s
                  << ", heater_W=" << newJob->heater_W;

              int W = 0;
              if (newJobIndex >= 0 &&
                  newJobIndex < static_cast<int>(jobWarmupTicks.size())) {
                W = jobWarmupTicks[newJobIndex];
                oss << " (warmup_ticks=" << W << ")";
              }
              oss << "\n";
            } else if (currentJob) {
              oss << "[job] tick=" << tickIndex
                  << " leaving job window; reverting to baseline (heater=0, beam off).\n";
            }
            log_msg(oss.str());

            currentJob      = newJob;
            currentJobIndex = newJobIndex;
          }

          if (currentJob) {
            job_tick_counter += 1;
          } else {
            job_tick_counter = 0;
          }

          // ---- 2) Decide heater demand, Fwafer, and mbe_active for this tick ----
          double heaterDemand_W = 1500.0;   // baseline if no jobs.txt
          double Fwafer_cmd     = std::numeric_limits<double>::quiet_NaN();
          double mbe_flag       = 0.0;     // 0 = beam off, 1 = beam on
          double target_T_K     = 300.0;   // idle baseline target

          if (!jobs.empty()) {
            if (currentJob) {
              Fwafer_cmd     = currentJob->Fwafer_cm2s;
              heaterDemand_W = fluxToHeaterPower(Fwafer_cmd);
              mbe_flag       = 1.0;

              // Target temperature derived directly from the job's flux.
              if (currentJob->Fwafer_cm2s > 0.0) {
                target_T_K = targetTempForFlux(currentJob->Fwafer_cm2s);
              }
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
              mbe_flag   = 0.0;
              target_T_K = 300.0; // idle target when no job is active
            }
          } else {
            // No jobs.txt: keep baseline heater and floor flux, beam off
            heaterDemand_W = 1500.0;
            if (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0) {
              Fwafer_cmd = FWAFFER_FLOOR_CM2S;
            } else {
              Fwafer_cmd = last_Fwafer_sent;
            }
            mbe_flag   = 0.0;
            target_T_K = 300.0; // idle baseline when running with no jobs.txt
          }

          // Inform EffusionCell of the "desired" crucible temperature implied
          // by the job's wafer flux. The cell will log target_temp_K alongside
          // act_temp_K so the discrepancy is visible in EffusionCell.csv.
          effCell.setTargetTempK(target_T_K);

          // Inform GrowthMonitor about the beam/job state for this tick
          int jobIndexForGrowth = currentJob ? currentJobIndex : -1;
          growth.setBeamState(jobIndexForGrowth,
                              mbe_flag > 0.5,
                              Fwafer_cmd);

          // ---- 3) Push Fwafer + mbe_active into params.inc when needed ----
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

            if (need_update) {
              std::ostringstream oss;
              oss << "[job] tick=" << tickIndex
                  << " update params.inc: Fwafer_cm2s=" << Fwafer_cmd
                  << ", mbe_active=" << mbe_flag << "\n";
              log_msg(oss.str());

              write_params_inc(Fwafer_cmd, mbe_flag, rank, args.inputDir, log_msg);
              wake.markDirtyReload();

              last_Fwafer_sent = Fwafer_cmd;
              last_mbe_sent    = mbe_flag;
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

          // ---- 6) After ticking, evaluate job health (under-flux + temp gate)
          if (currentJob && heaterDemand_W > 1e-6) {
            double P_actual = effCell.getLastHeatInputW();

            // 6a) Update RC temp proxy (same constants as EffusionCell::applyHeat).
            {
              const double C_J_PER_K = 1000.0;
              const double H_W_PER_K = 1.5;
              const double T_ENV_K   = 300.0;

              double net_W = P_actual - H_W_PER_K * (temp_proxy_K - T_ENV_K);
              double dT    = (net_W / C_J_PER_K) * dt;
              temp_proxy_K += dT;

              if (!std::isfinite(temp_proxy_K)) temp_proxy_K = T_ENV_K;
              if (temp_proxy_K < 0.0)           temp_proxy_K = 0.0;
            }

            // 6b) Determine if gates are armed for this job based on dynamic warm-up.
            int warmup_ticks_for_job = 0;
            if (currentJobIndex >= 0 &&
                currentJobIndex < static_cast<int>(jobWarmupTicks.size())) {
              warmup_ticks_for_job = jobWarmupTicks[currentJobIndex];
            }
            bool gates_armed = (job_tick_counter > warmup_ticks_for_job) &&
                               (target_T_K > 310.0); // ignore trivial ambient targets

            // 6c) Under-flux gate (power-based), only once gates are armed.
            double flux_ratio = 1.0;
            if (heaterDemand_W > 0.0) {
              flux_ratio = P_actual / heaterDemand_W;
            }
            if (!std::isfinite(flux_ratio)) {
              flux_ratio = 0.0;
            }

            if (gates_armed) {
              if (flux_ratio < MIN_FLUX_FRACTION) {
                underflux_streak += 1;
              } else {
                underflux_streak = 0;
              }
            } else {
              // Still in warm-up: do not accumulate under-flux streaks.
              underflux_streak = 0;
            }

            bool flux_gate_fail = (underflux_streak >= UNDERFLUX_LIMIT_TICKS);

            // 6d) Temperature gate: after warm-up, require T >= 95% of target.
            bool temp_gate_fail = false;

            if (gates_armed) {
              double temp_ratio = temp_proxy_K / target_T_K;
              if (!std::isfinite(temp_ratio)) {
                temp_ratio = 0.0;
              }

              if (temp_ratio < TEMP_TOLERANCE_FRACTION) {
                temp_miss_streak += 1;
              } else {
                temp_miss_streak = 0;
              }

              if (temp_miss_streak >= TEMP_FAIL_LIMIT_TICKS) {
                temp_gate_fail = true;
              }
            } else {
              // During warm-up, do not accumulate temperature misses.
              temp_miss_streak = 0;
            }

            // Push current streaks into globals so EffusionCell.csv can log them.
            g_underflux_streak_for_log = underflux_streak;
            g_temp_miss_streak_for_log = temp_miss_streak;

            // 6e) Abort if either gate fails, but only once per job.
            if ((flux_gate_fail || temp_gate_fail) &&
                currentJobIndex >= 0 &&
                currentJobIndex < static_cast<int>(jobAborted.size()) &&
                !jobAborted[currentJobIndex]) {

              jobAborted[currentJobIndex] = true;

              std::ostringstream joss;
              joss << "[job] tick=" << tickIndex
                   << " ABORTING job index " << currentJobIndex
                   << " due to ";
              if (flux_gate_fail && temp_gate_fail) {
                joss << "under-flux AND temperature-miss";
              } else if (flux_gate_fail) {
                joss << "under-flux";
              } else {
                joss << "temperature-miss";
              }
              joss << " (underflux_streak=" << underflux_streak
                   << ", temp_miss_streak=" << temp_miss_streak
                   << ", temp_proxy_K=" << temp_proxy_K
                   << ", target_T_K=" << target_T_K
                   << ", flux_ratio=" << flux_ratio
                   << ")\n";
              log_msg(joss.str());

              // Tell GrowthMonitor that this job got aborted (it will still
              // write the partial wafer at shutdown).
              growth.markJobAborted(currentJobIndex);

              // Mark failure for SimulationEngine.csv on this tick
              engine.markJobFailedThisTick();

              // Immediately tell SPARTA the beam is off.
              double F_for_abort = last_Fwafer_sent;
              if (!std::isfinite(F_for_abort) || F_for_abort <= 0.0) {
                F_for_abort = FWAFFER_FLOOR_CM2S;
              }
              write_params_inc(F_for_abort, 0.0, rank, args.inputDir, log_msg);
              wake.markDirtyReload();
              last_Fwafer_sent = F_for_abort;
              last_mbe_sent    = 0.0;

              // Reset job state so next tick we fall into "no job" path
              currentJob       = nullptr;
              currentJobIndex  = -1;
              underflux_streak = 0;
              temp_miss_streak = 0;
              job_tick_counter = 0;
              g_underflux_streak_for_log = 0;
              g_temp_miss_streak_for_log = 0;
              temp_proxy_K     = 300.0;
              last_heater_set  = std::numeric_limits<double>::quiet_NaN();
            }
          } else {
            // No active job or zero heater demand: clear temp streaks in logs.
            temp_miss_streak = 0;
            g_temp_miss_streak_for_log = 0;
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
