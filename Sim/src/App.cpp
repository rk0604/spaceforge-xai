// Sim/src/App.cpp
#include "App.hpp"

#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"
#include "WakeChamber.hpp"
#include "EffusionCell.hpp"
#include "orbit.hpp"
#include "Logger.hpp"
#include "GrowthMonitor.hpp"
#include "helpers.hpp"

// ---------------------------------------------------------------------
// Helper imports
// ---------------------------------------------------------------------
using SimHelpers::Args;
using SimHelpers::Job;
using SimHelpers::parse_args;
using SimHelpers::print_usage;
using SimHelpers::fluxToHeaterPower;
using SimHelpers::write_params_inc;
using SimHelpers::FWAFFER_FLOOR_CM2S;
using SimHelpers::LogFn;

// Globals from EffusionCell.cpp
extern int g_underflux_streak_for_log;
extern int g_temp_miss_streak_for_log;

// Global solar scale used by SolarArray
double g_orbit_solar_scale = 1.0;

// ---------------------------------------------------------------------
// Local helpers (unchanged)
// ---------------------------------------------------------------------
static double targetTempForFlux(double Fwafer_cm2s) {
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) return 300.0;

  const double F_low = 5e13, F_high = 1e14;
  const double T_low = 1100.0, T_high = 1300.0;

  double F = std::clamp(Fwafer_cm2s, F_low, F_high);
  double alpha = (std::log(F) - std::log(F_low)) /
                 (std::log(F_high) - std::log(F_low));
  alpha = std::clamp(alpha, 0.0, 1.0);
  return T_low + alpha * (T_high - T_low);
}

static int estimateWarmupTicksForFlux(double Fwafer_cm2s, double dt_s) {
  const double C = 1000.0, H = 1.5, Tenv = 300.0;
  if (!std::isfinite(dt_s) || dt_s <= 0.0) return 0;

  double P = fluxToHeaterPower(Fwafer_cm2s);
  if (P <= 0.0) return 0;

  double Tt = targetTempForFlux(Fwafer_cm2s);
  double Tss = Tenv + P / H;
  if (Tss <= Tenv) return 0;

  double Tgate = std::min(0.9 * Tt, 0.9 * Tss);
  double ratio = (Tgate - Tenv) / (Tss - Tenv);
  ratio = std::clamp(ratio, 0.0, 0.999);

  double tau = C / H;
  int ticks = static_cast<int>(std::ceil(-tau * std::log(1.0 - ratio) / dt_s));
  return std::clamp(ticks, 0, 60);
}

// ---------------------------------------------------------------------
// App implementation
// ---------------------------------------------------------------------
App::App(int argc, char** argv) : argc_(argc), argv_(argv) {}

int App::run() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Args args = parse_args(argc_, argv_);

  // ---------------- logging ----------------
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

  if (rank == 0) {
    const char* rid = std::getenv("RUN_ID");
    std::string run_id = rid ? rid : "norunid";
    std::string mode = args.mode.empty() ? "nomode" : args.mode;
    debugLog.open("sim_debug_" + run_id + "_" + mode + ".log", std::ios::app);
  }

  if (args.showHelp) {
    if (rank == 0) print_usage();
    return EXIT_SUCCESS;
  }

  // ---------------- clamps ----------------
  if (args.nticks <= 0) args.nticks = 500;
  if (args.dt <= 0.0) args.dt = 0.1;
  if (args.coupleEvery <= 0) args.coupleEvery = 10;
  if (args.spartaBlock <= 0) args.spartaBlock = 200;

  // ---------------- jobs ----------------
  std::vector<Job> jobs;
  if (rank == 0 && (args.mode == "wake" || args.mode == "dual" || args.mode == "legacy")) {
    std::ifstream jf(args.inputDir + "/jobs30.txt");
    if (jf) {
      Job j;
      while (jf >> j.start_tick >> j.end_tick >> j.Fwafer_cm2s >> j.heater_W) {
        if (j.end_tick < j.start_tick) std::swap(j.end_tick, j.start_tick);
        jobs.push_back(j);
      }
    }
  }

  int njobs = jobs.size();
  MPI_Bcast(&njobs, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> warmup;
  if (rank == 0) {
    warmup.resize(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i)
      warmup[i] = estimateWarmupTicksForFlux(jobs[i].Fwafer_cm2s, args.dt);
  }

  // ---------------- subsystems ----------------
  PowerBus bus;
  SolarArray solar(0.25, 8500.0);
  Battery battery;
  HeaterBank heater(2000.0);
  EffusionCell eff;
  GrowthMonitor growth(32);

  bus.setBattery(&battery);
  solar.setPowerBus(&bus);
  battery.setPowerBus(&bus);
  heater.setPowerBus(&bus);
  heater.setEffusionCell(&eff);
  growth.setPowerBus(&bus);
  growth.setIsLeader(rank == 0);
  growth.setNumJobs(jobs.size());

  SimulationEngine engine;
  engine.addSubsystem(&solar);
  engine.addSubsystem(&battery);
  engine.addSubsystem(&heater);
  engine.addSubsystem(&eff);
  engine.addSubsystem(&bus);
  engine.addSubsystem(&growth);

  engine.setTickStep(args.dt);
  engine.initialize();

  // ---------------- power-only mode ----------------
  if (args.mode == "power") {
    g_orbit_solar_scale = 1.0;
    for (int i = 0; i < args.nticks; ++i) {
      heater.setDemand(1500.0);
      growth.setBeamState(-1, false, 0.0);
      engine.tick();
      MPI_Barrier(MPI_COMM_WORLD);
    }
    engine.shutdown();
    return EXIT_SUCCESS;
  }

  // ---------------- wake / legacy / dual ----------------
  WakeChamber wake(MPI_COMM_WORLD, "WakeChamber");
  wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());

  OrbitModel orbit(300e3, args.dt, 0.0, 0.0);

  for (int i = 0; i < args.nticks; ++i) {
    if (rank == 0) {
      orbit.step();
      g_orbit_solar_scale = orbit.state().solar_scale;
    }

    engine.tick();

    if (i % args.coupleEvery == 0) {
      wake.runIfDirtyOrAdvance(args.spartaBlock);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  wake.shutdown();
  engine.shutdown();
  return EXIT_SUCCESS;
}
