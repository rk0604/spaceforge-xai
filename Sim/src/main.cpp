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
#include "SubstrateHeater.hpp"


// Bring helper types/functions into local scope
using SimHelpers::Args;
using SimHelpers::Job;
using SimHelpers::parse_args;
using SimHelpers::print_usage;
using SimHelpers::fluxToHeaterPower;
using SimHelpers::write_params_inc;
using SimHelpers::FWAFFER_FLOOR_CM2S;
using SimHelpers::LogFn;
using SimHelpers::PhaseCode;
using SimHelpers::phaseCodeName;
using SimHelpers::parseJobLine;
using SimHelpers::derivePhaseDurationTicks;
using SimHelpers::deriveLiveDepositionTicks;
using SimHelpers::isGrowthPhase;
using SimHelpers::isNonGrowthTimedPhase;
using SimHelpers::DEFAULT_IDLE_SUBSTRATE_TARGET_K;
using SimHelpers::DEFAULT_IDLE_EFFUSION_TARGET_K;
using SimHelpers::SourcePhasePolicy;
using SimHelpers::deriveSourcePhasePolicy;
using SimHelpers::resolveReferenceFluxForSourcePhase;
using SimHelpers::targetTempForFlux;

// Globals defined in EffusionCell.cpp so streaks show up in EffusionCell.csv.
extern int g_underflux_streak_for_log;
extern int g_temp_miss_streak_for_log;

// Global sunlight scale shared with SolarArray.cpp.
// In wake/dual/legacy mode this is updated each tick from OrbitModel.
// In power mode (no orbit), it stays at 1.0 (always sunlit).
double g_orbit_solar_scale = 1.0;

// ---------------------------------------------------------------------------
// Scheduler-facing per-row control intent
//
// This intent layer separates recipe policy from subsystem physics.
//
// Important design rule:
// - EffusionCell owns source thermal state evolution
// - SubstrateHeater owns substrate thermal state evolution
// - main.cpp owns which row currently controls the plant
// - helpers own phase semantics and phase-aware target selection
//
// This struct is intentionally richer than the older phase intent because
// beam-off phases such as SOURCE_DEGAS and SOAK may still require active
// source heating and source readiness.
// ---------------------------------------------------------------------------
struct PhaseControlIntent {
  bool beam_allowed = false;
  bool substrate_control_on = false;
  bool growth_like_phase = false;

  bool source_control_on = false;
  bool source_ready_required = false;
  bool flux_influences_source_target = false;
  bool source_may_idle = true;

  double effusion_target_K = DEFAULT_IDLE_EFFUSION_TARGET_K;
  double substrate_target_K = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
};


// ---------------------------------------------------------------------------
// Build explicit scheduler intent from the recipe row.
//
// This bridges row-level recipe semantics into concrete runtime control rules
// for source heating, substrate heating, readiness gating, and beam permission.
//
// Important:
// - beam permission does not imply source heating permission
// - source heating may be active during beam-off phases such as SOURCE_DEGAS
// - source readiness should only gate the phase when that phase actually
//   depends on source conditioning
// ---------------------------------------------------------------------------
static PhaseControlIntent derivePhaseIntent(double raw_job_flux_cm2s,
                                            int mbe_on,
                                            int substrate_on,
                                            PhaseCode phase_code,
                                            double explicit_substrate_target_K,
                                            double next_growth_flux_cm2s) {
  PhaseControlIntent out{};

  out.beam_allowed = (mbe_on != 0);
  out.substrate_control_on = (substrate_on != 0);
  out.growth_like_phase = isGrowthPhase(phase_code);

  const SourcePhasePolicy sourcePolicy =
      deriveSourcePhasePolicy(phase_code,
                              raw_job_flux_cm2s,
                              mbe_on,
                              next_growth_flux_cm2s);

  out.source_control_on = sourcePolicy.source_control_on;
  out.source_ready_required = sourcePolicy.source_ready_required;
  out.flux_influences_source_target = sourcePolicy.flux_influences_target;
  out.source_may_idle = sourcePolicy.source_may_idle;
  out.effusion_target_K = sourcePolicy.effusion_target_K;

  if (out.substrate_control_on &&
      std::isfinite(explicit_substrate_target_K) &&
      explicit_substrate_target_K > 0.0) {
    out.substrate_target_K = explicit_substrate_target_K;
  } else {
    out.substrate_target_K = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
  }

  return out;
}

// ---------------------------------------------------------------------------
// Estimate a notional warm-up time for a job based on its requested flux.
//
// Important:
// This is now DIAGNOSTIC ONLY.
//
// Under the new scheduler, warmup is no longer a grace-period hack that delays
// failure arming inside a requested job window. Instead, warmup/cooldown are
// real runtime scheduler states:
//
//   - Warming
//   - Cooling
//   - ThermalPrep
//
// with mbe_flag = 0 until the thermal subsystems are actually ready.
//
// So this estimate is retained only for logging / ML visibility / comparison,
// not for scheduling or deposition-duration accounting.
// ---------------------------------------------------------------------------

static int estimateWarmupTicksForFlux(double Fwafer_cm2s, double dt_s) {
  // Same RC constants as in the temp_proxy_K update below.
  const double C_J_PER_K = 800.0; // Was 1000.0
  const double H_W_PER_K = 0.8;  // Was 1.5
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

// ---------------------------------------------------------------------------
// Runtime scheduler model
//
// Important semantic shift:
// - jobs.txt start_tick now means "release / eligibility time"
// - requested beam-on duration is derived from (end_tick - start_tick)
// - actual prep start / actual deposition start / actual deposition end are
//   realized at runtime by the scheduler
//
// State-code mapping for TrainingState.csv:
//   0 = pending
//   1 = queued
//   2 = warming
//   3 = cooling
//   4 = thermal_prep
//   5 = live_deposition
//   6 = done
//   7 = aborted
//
// Prep-mode mapping for TrainingState.csv:
//   0 = none
//   1 = warmup
//   2 = cooldown
//   3 = thermal_prep_mixed
// ---------------------------------------------------------------------------
enum class JobRunState {
  Pending = 0,
  Queued = 1,
  Warming = 2,
  Cooling = 3,
  ThermalPrep = 4,
  LiveDeposition = 5,
  Done = 6,
  Aborted = 7
};

static const char* jobRunStateName(JobRunState s) {
  switch (s) {
    case JobRunState::Pending:        return "pending";
    case JobRunState::Queued:         return "queued";
    case JobRunState::Warming:        return "warming";
    case JobRunState::Cooling:        return "cooling";
    case JobRunState::ThermalPrep:    return "thermal_prep";
    case JobRunState::LiveDeposition: return "live_deposition";
    case JobRunState::Done:           return "done";
    case JobRunState::Aborted:        return "aborted";
    default:                          return "unknown";
  }
}

// readable prep labels in the schedule CSV
static const char* prepModeName(double prep_mode_code) {
  switch (static_cast<int>(prep_mode_code)) {
    case 0: return "none";
    case 1: return "warmup";
    case 2: return "cooldown";
    case 3: return "thermal_prep_mixed";
    default: return "unknown";
  }
}

static double jobRunStateCode(JobRunState s) {
  return static_cast<double>(static_cast<int>(s));
}


struct RuntimeJobState {
  // Immutable requested plan
  int    requested_start_tick = 0;
  int    requested_end_tick = -1;

  // Total scheduled recipe time represented by the row.
  int    requested_phase_duration_ticks = 0;

  // Beam-on deposition time owed by growth-like rows only.
  int    requested_live_duration_ticks = 0;

  double requested_flux_cm2s = 0.0;
  double requested_heater_cap_W = 0.0;

  int    requested_mbe_on = 0;
  int    requested_substrate_on = 0;
  PhaseCode requested_phase_code = PhaseCode::IDLE;
  double requested_substrate_target_K = DEFAULT_IDLE_SUBSTRATE_TARGET_K;

  // Runtime execution state
  JobRunState state = JobRunState::Pending;
  bool released_to_queue = false;
  bool done = false;
  bool aborted = false;

  // Latched execution semantics for growth-like phases.
  bool has_started_live_execution = false;

  // Latched when a growth job loses required hold after live execution started.
  bool live_execution_hold_faulted = false;

  // Realized timeline
  int actual_queue_enter_tick = -1;
  int actual_thermal_prep_start_tick = -1;
  int actual_warmup_start_tick = -1;
  int actual_cooldown_start_tick = -1;

  // Realized start and end of the recipe phase once thermally executable.
  int actual_phase_start_tick = -1;
  int actual_phase_end_tick = -1;

  // Realized beam-on deposition interval for growth-like rows.
  int actual_deposition_start_tick = -1;
  int actual_deposition_end_tick = -1;

  // Total recipe phase accounting
  int phase_ticks_completed = 0;
  int remaining_phase_ticks = 0;

  // Guaranteed beam-on accounting
  int live_ticks_completed = 0;
  int remaining_live_ticks = 0;
};

// ---------------------------------------------------------------------------
// Find the next positive growth-like flux after the currently controlling row.
//
// This gives beam-off pre-growth phases such as SOURCE_DEGAS and SOAK a
// physically motivated anchor for source thermal conditioning when the current
// row itself does not carry a meaningful growth flux.
//
// If no future growth-like row exists, this returns 0.0 and the helper-side
// source policy will fall back to its built-in default anchors.
// ---------------------------------------------------------------------------
static double findNextGrowthFluxCm2s(const std::vector<RuntimeJobState>& runtimeJobs,
                                     int currentIndex) {
  if (currentIndex < 0) {
    return 0.0;
  }

  for (std::size_t idx = static_cast<std::size_t>(currentIndex + 1);
       idx < runtimeJobs.size();
       ++idx) {
    const RuntimeJobState& candidate = runtimeJobs[idx];

    const bool growth_like = isGrowthPhase(candidate.requested_phase_code);
    const bool beam_enabled = (candidate.requested_mbe_on != 0);
    const bool flux_valid =
        std::isfinite(candidate.requested_flux_cm2s) &&
        candidate.requested_flux_cm2s > 0.0;

    if (growth_like && beam_enabled && flux_valid) {
      return candidate.requested_flux_cm2s;
    }
  }

  return 0.0;
}

// earlier there was a deriveRequestedDurationTicks function but it was deleted because it collapses zero-flux rows to duration 0.
// --------------------------------------------------------- main loop ---------------------------------------------------------
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

    // ------------------------------------------------------------------------
  // Per-rank progress logger for MPI debugging.
  // Writes one file per rank so we can see where each rank stops.
  // ------------------------------------------------------------------------
  std::ofstream rankProgressLog;
  {
    const char* env_run_id = std::getenv("RUN_ID");
    std::string run_id     = env_run_id ? env_run_id : "norunid";
    std::string mode_tag   = args.mode.empty() ? "nomode" : args.mode;

    std::string rank_filename =
        "sim_rank_progress_r" + std::to_string(rank) +
        "_" + run_id + "_" + mode_tag + ".log";

    rankProgressLog.open(rank_filename, std::ios::out | std::ios::app);
    if (!rankProgressLog) {
      std::cerr << "[warn] rank " << rank
                << " failed to open " << rank_filename << " for writing.\n";
      std::cerr.flush();
    }
  }

  auto log_rank_progress = [&](int tick, const std::string& phase) {
    std::ostringstream oss;
    oss << "[rank " << rank << "] tick=" << tick
        << " phase=" << phase << "\n";

    std::cerr << oss.str();
    std::cerr.flush();

    if (rankProgressLog.is_open()) {
      rankProgressLog << oss.str();
      rankProgressLog.flush();
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
        const std::string jobsPath = args.inputDir + "/test.txt";
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

            Job j{};
            std::string parseErr;
            if (!parseJobLine(line, j, parseErr)) {
              std::ostringstream oss;
              oss << "[warn] jobs.txt line " << lineno
                  << " malformed, skipping: " << line
                  << " | reason: " << parseErr << "\n";
              log_msg(oss.str());
              continue;
            }
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
                << " cm^-2 s^-1, heater_cap=" << j.heater_W
                << " W, mbe_on=" << j.mbe_on
                << ", substrate_on=" << j.substrate_on
                << ", phase=" << phaseCodeName(j.phase_code)
                << ", substrate_target_K=" << j.substrate_target_K
                << "\n";
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

    // Electrical/power subsystems (independent of SPARTA)
    PowerBus bus;
    const double SOLAR_EFFICIENCY   = 0.25;
    const double SOLAR_BASE_INPUT_W = 30000.0; // example boost

    SolarArray solar(SOLAR_EFFICIENCY, SOLAR_BASE_INPUT_W);
    Battery battery;

    bus.setBattery(&battery);
    // Bigger heater: can draw up to 2 kW from the bus.
    HeaterBank    heater(/*maxDraw=*/5000.0);
    EffusionCell  effCell;
    GrowthMonitor growth(/*gridN=*/32);
    SubstrateHeater substrateHeater(/*maxPowerW=*/3000.0, /*wafer_radius_m=*/0.15);

  
    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);
    heater.setEffusionCell(&effCell);     // Heater warms the effusion cell
    heater.setSubstrateHeater(&substrateHeater); // heater warms up the substrate as well
    growth.setPowerBus(&bus);             // Film/growth monitor draws instrument power

    // GrowthMonitor should only log + write CSV on leader, but engine tick
    // will be called on all ranks.
    growth.setIsLeader(rank == 0);
    substrateHeater.setIsLeader(rank == 0);
    growth.setNumJobs(njobs);

    /**
     * old order 
        engine.addSubsystem(&solar);      // 1) power source
        engine.addSubsystem(&battery);    // 2) storage update
        engine.addSubsystem(&heater);     // 3) power load
        engine.addSubsystem(&substrateHeater);
        engine.addSubsystem(&effCell);    // 4) heat response (after heater!)
        engine.addSubsystem(&bus);        // 5) bookkeeping on power totals
        engine.addSubsystem(&growth);     // 6) sensors/aux
     * 
     */

    SimulationEngine engine;

    // 1) Solar generates power and adds it to the bus early in the tick
    engine.addSubsystem(&solar);

    // 2) Loads draw power during the tick
    engine.addSubsystem(&heater);

    // 3) These must tick AFTER heater because heater.applyHeat() happens inside HeaterBank::tick()
    engine.addSubsystem(&substrateHeater);
    engine.addSubsystem(&effCell);

    // 4) GrowthMonitor draws instrument power (so it MUST be before bus bookkeeping)
    engine.addSubsystem(&growth);

    // 5) Bus bookkeeping MUST be after all producers/consumers have run
    engine.addSubsystem(&bus);

    // 6) Battery logs LAST so it reflects the post-bus charge/discharge state for the tick
    engine.addSubsystem(&battery);


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
        }

        
        heater.setEffusionDemand(1500.0);
        heater.setSubstrateDemand(0.0);
        heater.setPrioritySubstrate(false);
        // No jobs in power-only mode -> growth monitor gets jobIndex=-1, mbeOff.
        growth.setBeamState(-1, false, 0.0);
        engine.tick();
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
        // Seed SPARTA with a legal positive mixture flux.
        // This is NOT a statement that physical deposition is active.
        double first_raw_flux = jobs.front().Fwafer_cm2s;
        if (std::isfinite(first_raw_flux) && first_raw_flux > 0.0) {
          initial_Fwafer = first_raw_flux;
        } else {
          initial_Fwafer = FWAFFER_FLOOR_CM2S;
        }
      }

      // Broadcast initial flux to all ranks so they agree
      MPI_Bcast(&initial_Fwafer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Beam off initially (mbe_active = 0), but flux positive so mixture is legal
      double last_Fwafer_sent = std::numeric_limits<double>::quiet_NaN();
      double last_mbe_sent    = std::numeric_limits<double>::quiet_NaN(); 
      
      log_rank_progress(0, "before-initial-write_params_inc");
      write_params_inc(initial_Fwafer, 0.0, rank, args.inputDir, log_msg);
      log_rank_progress(0, "after-initial-write_params_inc");

      last_Fwafer_sent = initial_Fwafer;
      last_mbe_sent    = 0.0;

      if (rank == 0) {
        log_msg("[info] Constructing WakeChamber and calling wake.init(...)\n");
      }

      WakeChamber wake(MPI_COMM_WORLD, "WakeChamber");
      log_rank_progress(0, "before-wake-init");
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());
      log_rank_progress(0, "after-wake-init");


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
      // -------------------------------------------------------------------
      // Single-owner runtime scheduler state (leader only)
      //
      // Important:
      // - jobs.txt controls only release/eligibility time
      // - exactly one controlling job may own the effusion/substrate targets
      //   and SPARTA beam semantics at a time
      // - later released jobs wait in queue until the current controller is
      //   done or aborted
      // -------------------------------------------------------------------
      std::vector<RuntimeJobState> runtimeJobs;
      runtimeJobs.reserve(jobs.size());

      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const Job& j = jobs[i];

        RuntimeJobState rj;

        // Copy the immutable schedule request into runtime state.
        rj.requested_start_tick         = j.start_tick;
        rj.requested_end_tick           = j.end_tick;
        rj.requested_phase_duration_ticks = derivePhaseDurationTicks(j);
        rj.requested_live_duration_ticks  = deriveLiveDepositionTicks(j);

        rj.requested_flux_cm2s          = j.Fwafer_cm2s;
        rj.requested_heater_cap_W       = j.heater_W;
        rj.requested_mbe_on             = j.mbe_on;
        rj.requested_substrate_on       = j.substrate_on;
        rj.requested_phase_code         = j.phase_code;
        rj.requested_substrate_target_K = j.substrate_target_K;

        // Phase time and live deposition time are tracked separately.
        // Non-growth timed phases can now persist even when live deposition owed is zero.
        rj.remaining_phase_ticks = rj.requested_phase_duration_ticks;
        rj.remaining_live_ticks  = rj.requested_live_duration_ticks;

        runtimeJobs.push_back(rj);
      }

      // Which job currently owns the thermal targets / beam state?
      int controllingJobIndex = -1;

      // Cached logging values
      double last_heater_set    = std::numeric_limits<double>::quiet_NaN();
      double last_effusion_set  = std::numeric_limits<double>::quiet_NaN();
      double last_substrate_set = std::numeric_limits<double>::quiet_NaN();

      // Health tracking for the CURRENT controlling deposition job only.
      int   underflux_streak           = 0;
      int   temp_miss_streak           = 0;
      const int    UNDERFLUX_LIMIT_TICKS   = 20;
      const double MIN_FLUX_FRACTION       = 0.90;
      const int    TEMP_FAIL_LIMIT_TICKS   = 20;
      const double TEMP_TOLERANCE_FRACTION = 0.85;

      // RC temp proxy (mirrors EffusionCell RC constants) used for a
      // conservative temperature-health gate.
      double temp_proxy_K = 300.0;

            auto emitPostAccountingLogs =
          [&](int tickIndex,
              double t_phys,
              int controllingJobIndex_snapshot,
              double raw_job_flux_cm2s_snapshot,
              double sparta_flux_cm2s_snapshot,
              bool deposition_requested_snapshot,
              double effusionDemand_W_snapshot,
              double substrateDemand_W_snapshot,
              double target_T_K_snapshot,
              double scheduler_substrate_target_K_log_snapshot,
              double phase_ready_for_execution_log_snapshot,
              double mbe_flag_snapshot,
              bool effusion_ready_snapshot,
              bool wafer_ready_snapshot) {

        int log_job_index = controllingJobIndex_snapshot;

        double scheduler_state_code_log = jobRunStateCode(JobRunState::Pending);
        double prep_mode_code = 0.0;

        double requested_start_tick_log           = -1.0;
        double requested_end_tick_log             = -1.0;
        double requested_phase_duration_ticks_log = 0.0;
        double requested_live_duration_ticks_log  = 0.0;

        double actual_queue_enter_tick_log        = -1.0;
        double actual_thermal_prep_start_log      = -1.0;
        double actual_warmup_start_tick_log       = -1.0;
        double actual_cooldown_start_tick_log     = -1.0;
        double actual_phase_start_tick_log        = -1.0;
        double actual_phase_end_tick_log          = -1.0;
        double actual_deposition_start_tick_log   = -1.0;
        double actual_deposition_end_tick_log     = -1.0;

        double phase_ticks_completed_log          = 0.0;
        double remaining_phase_ticks_log          = 0.0;
        double live_ticks_completed_log           = 0.0;
        double remaining_live_ticks_log           = 0.0;
        double delay_from_requested_start_log     = 0.0;

        double phase_code_log                     = static_cast<double>(static_cast<int>(PhaseCode::IDLE));
        double mbe_on_log                         = 0.0;
        double substrate_on_log                   = 0.0;

        double queued_active       = 0.0;
        double warmup_active       = 0.0;
        double cooldown_active     = 0.0;
        double thermal_prep_active = 0.0;
        double live_active         = 0.0;
        double done_active         = 0.0;
        double aborted_active      = 0.0;

        std::string phase_name_log = "NONE";

        if (log_job_index >= 0 &&
            log_job_index < static_cast<int>(runtimeJobs.size())) {
          const RuntimeJobState& rj = runtimeJobs[log_job_index];

          scheduler_state_code_log = jobRunStateCode(rj.state);

          requested_start_tick_log           = static_cast<double>(rj.requested_start_tick);
          requested_end_tick_log             = static_cast<double>(rj.requested_end_tick);
          requested_phase_duration_ticks_log = static_cast<double>(rj.requested_phase_duration_ticks);
          requested_live_duration_ticks_log  = static_cast<double>(rj.requested_live_duration_ticks);

          actual_queue_enter_tick_log        = static_cast<double>(rj.actual_queue_enter_tick);
          actual_thermal_prep_start_log      = static_cast<double>(rj.actual_thermal_prep_start_tick);
          actual_warmup_start_tick_log       = static_cast<double>(rj.actual_warmup_start_tick);
          actual_cooldown_start_tick_log     = static_cast<double>(rj.actual_cooldown_start_tick);
          actual_phase_start_tick_log        = static_cast<double>(rj.actual_phase_start_tick);
          actual_phase_end_tick_log          = static_cast<double>(rj.actual_phase_end_tick);
          actual_deposition_start_tick_log   = static_cast<double>(rj.actual_deposition_start_tick);
          actual_deposition_end_tick_log     = static_cast<double>(rj.actual_deposition_end_tick);

          phase_ticks_completed_log          = static_cast<double>(rj.phase_ticks_completed);
          remaining_phase_ticks_log          = static_cast<double>(rj.remaining_phase_ticks);
          live_ticks_completed_log           = static_cast<double>(rj.live_ticks_completed);
          remaining_live_ticks_log           = static_cast<double>(rj.remaining_live_ticks);

          phase_code_log                     = static_cast<double>(static_cast<int>(rj.requested_phase_code));
          mbe_on_log                         = static_cast<double>(rj.requested_mbe_on);
          substrate_on_log                   = static_cast<double>(rj.requested_substrate_on);
          phase_name_log                     = phaseCodeName(rj.requested_phase_code);

          if (rj.actual_phase_start_tick >= 0) {
            delay_from_requested_start_log =
                static_cast<double>(rj.actual_phase_start_tick - rj.requested_start_tick);
          } else if (rj.actual_deposition_start_tick >= 0) {
            delay_from_requested_start_log =
                static_cast<double>(rj.actual_deposition_start_tick - rj.requested_start_tick);
          } else {
            delay_from_requested_start_log =
                static_cast<double>(tickIndex - rj.requested_start_tick);
          }

          queued_active       = (rj.state == JobRunState::Queued) ? 1.0 : 0.0;
          warmup_active       = (rj.state == JobRunState::Warming) ? 1.0 : 0.0;
          cooldown_active     = (rj.state == JobRunState::Cooling) ? 1.0 : 0.0;
          thermal_prep_active = (rj.state == JobRunState::ThermalPrep) ? 1.0 : 0.0;
          live_active         = (rj.state == JobRunState::LiveDeposition) ? 1.0 : 0.0;
          done_active         = (rj.state == JobRunState::Done) ? 1.0 : 0.0;
          aborted_active      = (rj.state == JobRunState::Aborted) ? 1.0 : 0.0;

          if (rj.state == JobRunState::Warming) {
            prep_mode_code = 1.0;
          } else if (rj.state == JobRunState::Cooling) {
            prep_mode_code = 2.0;
          } else if (rj.state == JobRunState::ThermalPrep &&
                     phase_ready_for_execution_log_snapshot <= 0.5) {
            prep_mode_code = 3.0;
          } else {
            prep_mode_code = 0.0;
          }
        }

        Logger::instance().log_wide(
          "ScheduleState",
          tickIndex,
          t_phys,
          {
            "controlling_job_index",
            "scheduler_state_code",
            "prep_mode_code",

            "requested_start_tick",
            "requested_end_tick",
            "requested_phase_duration_ticks",
            "requested_live_duration_ticks",

            "actual_queue_enter_tick",
            "actual_thermal_prep_start_tick",
            "actual_warmup_start_tick",
            "actual_cooldown_start_tick",
            "actual_phase_start_tick",
            "actual_phase_end_tick",
            "actual_deposition_start_tick",
            "actual_deposition_end_tick",

            "phase_ticks_completed",
            "remaining_phase_ticks",
            "live_ticks_completed",
            "remaining_live_ticks",
            "delay_from_requested_start",

            "phase_code",
            "mbe_on",
            "substrate_on",
            "phase_ready_for_execution",

            "queued_active",
            "warmup_active",
            "cooldown_active",
            "thermal_prep_active",
            "live_active",
            "done_active",
            "aborted_active",

            "deposition_requested",
            "mbe_flag"
          },
          {
            static_cast<double>(log_job_index),
            scheduler_state_code_log,
            prep_mode_code,

            requested_start_tick_log,
            requested_end_tick_log,
            requested_phase_duration_ticks_log,
            requested_live_duration_ticks_log,

            actual_queue_enter_tick_log,
            actual_thermal_prep_start_log,
            actual_warmup_start_tick_log,
            actual_cooldown_start_tick_log,
            actual_phase_start_tick_log,
            actual_phase_end_tick_log,
            actual_deposition_start_tick_log,
            actual_deposition_end_tick_log,

            phase_ticks_completed_log,
            remaining_phase_ticks_log,
            live_ticks_completed_log,
            remaining_live_ticks_log,
            delay_from_requested_start_log,

            phase_code_log,
            mbe_on_log,
            substrate_on_log,
            phase_ready_for_execution_log_snapshot,

            queued_active,
            warmup_active,
            cooldown_active,
            thermal_prep_active,
            live_active,
            done_active,
            aborted_active,

            deposition_requested_snapshot ? 1.0 : 0.0,
            mbe_flag_snapshot
          }
        );

        Logger::instance().log_wide(
          "ScheduleStateText",
          tickIndex,
          t_phys,
          std::vector<std::string>{
            "scheduler_state_name",
            "prep_mode_name",
            "phase_name"
          },
          std::vector<std::string>{
            jobRunStateName(static_cast<JobRunState>(static_cast<int>(scheduler_state_code_log))),
            prepModeName(prep_mode_code),
            phase_name_log
          }
        );

        Logger::instance().log_wide(
          "ProcessState",
          tickIndex,
          t_phys,
          {
            "controlling_job_index",
            "phase_code",
            "mbe_on",
            "substrate_on",

            "raw_job_flux_cm2s",
            "sparta_flux_cm2s",
            "deposition_requested",
            "phase_ready_for_execution",

            "effusionDemand_W",
            "substrateDemand_W",

            "effusion_temp_K",
            "effusion_target_K",
            "substrate_temp_K",
            "substrate_target_K",

            "effusion_ready",
            "wafer_ready",
            "mbe_flag",

            "effusion_delivered_W",
            "substrate_delivered_W",

            "underflux_streak",
            "temp_miss_streak"
          },
          {
            static_cast<double>(log_job_index),
            phase_code_log,
            mbe_on_log,
            substrate_on_log,

            raw_job_flux_cm2s_snapshot,
            sparta_flux_cm2s_snapshot,
            deposition_requested_snapshot ? 1.0 : 0.0,
            phase_ready_for_execution_log_snapshot,

            effusionDemand_W_snapshot,
            substrateDemand_W_snapshot,

            effCell.getTemperatureK(),
            target_T_K_snapshot,
            substrateHeater.substrateTempK(),
            scheduler_substrate_target_K_log_snapshot,

            effusion_ready_snapshot ? 1.0 : 0.0,
            wafer_ready_snapshot ? 1.0 : 0.0,
            mbe_flag_snapshot,

            effCell.getLastHeatInputW(),
            substrateHeater.deliveredPowerW(),

            static_cast<double>(underflux_streak),
            static_cast<double>(temp_miss_streak)
          }
        );

        {
          std::ostringstream oss;
          oss << "[sched-state] tick=" << tickIndex
              << " job=" << log_job_index
              << " state=" << jobRunStateName(static_cast<JobRunState>(static_cast<int>(scheduler_state_code_log)))
              << " prep=" << prepModeName(prep_mode_code)
              << " remaining_live=" << remaining_live_ticks_log
              << " mbe=" << mbe_flag_snapshot << "\n";
          log_msg(oss.str());
        }
      };

      const int NTICKS = args.nticks;
      constexpr double PI_MAIN = 3.141592653589793;

      for (int i = 0; i < NTICKS; ++i) {
        const int tickIndex = i + 1;
        const double t_phys = tickIndex * dt;
        log_rank_progress(tickIndex, "loop-top");

        // ---------------- leader: orbit, job schedule, per-tick harness -----
        if (isLeader) {
          int logJobIndexAfterAccounting = controllingJobIndex;
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

// deleted stuff here!!!

                    // ---- 1) Release newly eligible jobs into the queue ----
          for (std::size_t idx = 0; idx < runtimeJobs.size(); ++idx) {
            RuntimeJobState& rj = runtimeJobs[idx];

            if (rj.state == JobRunState::Pending && tickIndex >= rj.requested_start_tick) {
              rj.state = JobRunState::Queued;
              rj.released_to_queue = true;
              rj.actual_queue_enter_tick = tickIndex;

              std::ostringstream oss;
              oss << "[sched] tick=" << tickIndex
                  << " released job " << idx
                  << " into queue"
                  << " (requested_start=" << rj.requested_start_tick
                  << ", requested_phase_duration=" << rj.requested_phase_duration_ticks
                  << ", requested_live_duration=" << rj.requested_live_duration_ticks
                  << ", requested_flux=" << rj.requested_flux_cm2s
                  << ")\n";
              log_msg(oss.str());
            }
          }

          // ---- 2) If nobody owns control, promote the earliest queued job ----
          if (controllingJobIndex < 0) {
            for (std::size_t idx = 0; idx < runtimeJobs.size(); ++idx) {
              RuntimeJobState& rj = runtimeJobs[idx];
              if (rj.state == JobRunState::Queued && !rj.done && !rj.aborted) {
                controllingJobIndex = static_cast<int>(idx);

                // Reset controller-local failure tracking on ownership transfer.
                underflux_streak = 0;
                temp_miss_streak = 0;
                g_underflux_streak_for_log = 0;
                g_temp_miss_streak_for_log = 0;
                temp_proxy_K = 300.0;

                // Reset latched execution fault semantics for the new controller.
                rj.has_started_live_execution = false;
                rj.live_execution_hold_faulted = false;

                std::ostringstream oss;
                oss << "[sched] tick=" << tickIndex
                    << " job " << idx << " took control"
                    << " (requested_start=" << rj.requested_start_tick
                    << ", requested_end=" << rj.requested_end_tick
                    << ", requested_phase_duration=" << rj.requested_phase_duration_ticks
                    << ", requested_live_duration=" << rj.requested_live_duration_ticks
                    << ", remaining_live_ticks=" << rj.remaining_live_ticks
                    << ")\n";
                log_msg(oss.str());
                break;
              }
            }
          }

          // ---- 3) Decide scheduler state, heater demand, flux, and mbe flag ----
          double target_T_K           = DEFAULT_IDLE_EFFUSION_TARGET_K;
          double scheduler_substrate_target_K_log = DEFAULT_IDLE_SUBSTRATE_TARGET_K;

          double raw_job_flux_cm2s    = 0.0;
          double sparta_flux_cm2s     = FWAFFER_FLOOR_CM2S;
          bool   deposition_requested = false;

          double effusionDemand_W     = 0.0;
          double substrateDemand_W    = 0.0;

          double mbe_flag             = 0.0;

          bool   wafer_ready          = true;
          bool   effusion_ready       = true;

          int    jobIndexForGrowth    = -1;
          double growth_flux_cm2s     = 0.0;

          double queued_active        = 0.0;
          double warmup_active        = 0.0;
          double cooldown_active      = 0.0;
          double thermal_prep_active  = 0.0;
          double live_active          = 0.0;
          double done_active          = 0.0;
          double aborted_active       = 0.0;
          double prep_mode_code       = 0.0;

          double requested_start_tick_log            = -1.0;
          double requested_end_tick_log              = -1.0;
          double requested_phase_duration_ticks_log  = 0.0;
          double requested_live_duration_ticks_log   = 0.0;

          double actual_queue_enter_tick_log         = -1.0;
          double actual_thermal_prep_start_log       = -1.0;
          double actual_warmup_start_tick_log        = -1.0;
          double actual_cooldown_start_tick_log      = -1.0;
          double actual_phase_start_tick_log         = -1.0;
          double actual_phase_end_tick_log           = -1.0;
          double actual_deposition_start_tick_log    = -1.0;
          double actual_deposition_end_tick_log      = -1.0;

          double phase_ticks_completed_log           = 0.0;
          double remaining_phase_ticks_log           = 0.0;
          double live_ticks_completed_log            = 0.0;
          double remaining_live_ticks_log            = 0.0;

          double delay_from_requested_start_log      = 0.0;
          double scheduler_state_code_log            = jobRunStateCode(JobRunState::Pending);

          double phase_code_log                      = static_cast<double>(static_cast<int>(PhaseCode::IDLE));
          double mbe_on_log                          = 0.0;
          double substrate_on_log                    = 0.0;
          double phase_ready_for_execution_log       = 0.0;

          if (controllingJobIndex >= 0) {
            RuntimeJobState& rj = runtimeJobs[controllingJobIndex];
            jobIndexForGrowth = controllingJobIndex;

            requested_start_tick_log          = static_cast<double>(rj.requested_start_tick);
            requested_end_tick_log            = static_cast<double>(rj.requested_end_tick);
            requested_phase_duration_ticks_log = static_cast<double>(rj.requested_phase_duration_ticks);
            requested_live_duration_ticks_log  = static_cast<double>(rj.requested_live_duration_ticks);

            actual_queue_enter_tick_log       = static_cast<double>(rj.actual_queue_enter_tick);
            actual_thermal_prep_start_log     = static_cast<double>(rj.actual_thermal_prep_start_tick);
            actual_warmup_start_tick_log      = static_cast<double>(rj.actual_warmup_start_tick);
            actual_cooldown_start_tick_log    = static_cast<double>(rj.actual_cooldown_start_tick);
            actual_phase_start_tick_log       = static_cast<double>(rj.actual_phase_start_tick);
            actual_phase_end_tick_log         = static_cast<double>(rj.actual_phase_end_tick);
            actual_deposition_start_tick_log  = static_cast<double>(rj.actual_deposition_start_tick);
            actual_deposition_end_tick_log    = static_cast<double>(rj.actual_deposition_end_tick);

            phase_ticks_completed_log         = static_cast<double>(rj.phase_ticks_completed);
            remaining_phase_ticks_log         = static_cast<double>(rj.remaining_phase_ticks);
            live_ticks_completed_log          = static_cast<double>(rj.live_ticks_completed);
            remaining_live_ticks_log          = static_cast<double>(rj.remaining_live_ticks);

            phase_code_log                    = static_cast<double>(static_cast<int>(rj.requested_phase_code));
            mbe_on_log                        = static_cast<double>(rj.requested_mbe_on);
            substrate_on_log                  = static_cast<double>(rj.requested_substrate_on);

            raw_job_flux_cm2s = rj.requested_flux_cm2s;
            if (!std::isfinite(raw_job_flux_cm2s) || raw_job_flux_cm2s < 0.0) {
              raw_job_flux_cm2s = 0.0;
            }

            // Determine the next growth-like flux anchor so beam-off source
            // conditioning phases can still choose a physically motivated source
            // target even when they are not themselves depositing material.
            //
            // Examples:
            // - SOURCE_DEGAS can heat the source above ambient with mbe_flag = 0
            // - SOAK can hold near the upcoming growth condition without beam-on
            const double next_growth_flux_cm2s =
                findNextGrowthFluxCm2s(runtimeJobs, controllingJobIndex);

            // Build explicit scheduler intent from the active recipe row.
            //
            // Important semantic separation:
            // - beam permission controls whether deposition may occur
            // - source control controls whether the source should thermally
            //   condition or hold at an elevated target
            // - substrate control controls whether the wafer heater should aim
            //   for the row's substrate target
            const PhaseControlIntent phaseIntent = derivePhaseIntent(
                raw_job_flux_cm2s,
                rj.requested_mbe_on,
                rj.requested_substrate_on,
                rj.requested_phase_code,
                rj.requested_substrate_target_K,
                next_growth_flux_cm2s
            );

            scheduler_substrate_target_K_log = phaseIntent.substrate_target_K;

            // Physical deposition is only requested for beam-enabled growth-like
            // phases with positive row flux. Beam-off thermal phases may still
            // actively heat the source, but they must not open the beam.
            deposition_requested =
                phaseIntent.beam_allowed &&
                phaseIntent.growth_like_phase &&
                raw_job_flux_cm2s > 0.0;

            // SPARTA still needs a legal positive mixture value even when the
            // physical beam is off. The scheduler keeps those semantics intact:
            // - sparta_flux_cm2s stays legally positive for mixture stability
            // - actual deposition remains controlled strictly by mbe_flag
            sparta_flux_cm2s =
                deposition_requested ? raw_job_flux_cm2s : FWAFFER_FLOOR_CM2S;

            // Apply the phase-aware source target before computing heater demand.
            //
            // This is the key fix for SOURCE_DEGAS and other beam-off source
            // conditioning phases. A phase can now request active source heating
            // without implying beam-on deposition.

            target_T_K = phaseIntent.effusion_target_K;
            effCell.setTargetTempK(target_T_K);
            substrateHeater.setJobState(
                controllingJobIndex,
                /*job_active=*/true,
                raw_job_flux_cm2s,
                phaseIntent.substrate_control_on,
                phaseIntent.substrate_target_K
            );

            {
              std::ostringstream oss;
              oss << "[phase-intent] tick=" << tickIndex
                  << " job=" << controllingJobIndex
                  << " phase=" << phaseCodeName(rj.requested_phase_code)
                  << " source_control_on=" << (phaseIntent.source_control_on ? 1 : 0)
                  << " source_ready_required=" << (phaseIntent.source_ready_required ? 1 : 0)
                  << " beam_allowed=" << (phaseIntent.beam_allowed ? 1 : 0)
                  << " growth_like_phase=" << (phaseIntent.growth_like_phase ? 1 : 0)
                  << " raw_flux_cm2s=" << raw_job_flux_cm2s
                  << " next_growth_flux_cm2s=" << next_growth_flux_cm2s
                  << " effusion_target_K=" << target_T_K
                  << " substrate_target_K=" << scheduler_substrate_target_K_log
                  << "\n";
              log_msg(oss.str());
            }

            // A row is complete only when its total scheduled phase time is done.
            // This prevents zero-flux thermal phases from collapsing immediately.
            if (rj.remaining_phase_ticks <= 0) {
              rj.done = true;
              rj.state = JobRunState::Done;
              rj.actual_phase_end_tick = tickIndex;

              if (rj.actual_deposition_start_tick >= 0 && rj.actual_deposition_end_tick < 0) {
                rj.actual_deposition_end_tick = tickIndex;
              }

              std::ostringstream oss;
              oss << "[sched] tick=" << tickIndex
                  << " job " << controllingJobIndex
                  << " completed because remaining_phase_ticks=0\n";
              log_msg(oss.str());

              logJobIndexAfterAccounting = controllingJobIndex;
              
              controllingJobIndex = -1;
              substrateHeater.setJobState(
                  -1,
                  /*job_active=*/false,
                  0.0,
                  false,
                  DEFAULT_IDLE_SUBSTRATE_TARGET_K
              );

              raw_job_flux_cm2s = 0.0;
              deposition_requested = false;
              sparta_flux_cm2s = (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0)
                                   ? FWAFFER_FLOOR_CM2S
                                   : last_Fwafer_sent;
              target_T_K = DEFAULT_IDLE_EFFUSION_TARGET_K;
              effCell.setTargetTempK(target_T_K);
              effusionDemand_W = 0.0;
              substrateDemand_W = substrateHeater.computePowerRequestW();
              wafer_ready = true;
              effusion_ready = true;
              mbe_flag = 0.0;
              done_active = 1.0;
              scheduler_state_code_log = jobRunStateCode(JobRunState::Done);
            } else {
                            const double EFF_T_ENV_K       = DEFAULT_IDLE_EFFUSION_TARGET_K;
              const double EFF_H_W_PER_K     = 0.8;
              const double EFF_KP_W_PER_K    = 8.0;
              const double EFF_DEFAULT_MAX_W = 2000.0;

              const double eff_temp_K = effCell.getTemperatureK();
              const double eff_temp_error_K = target_T_K - eff_temp_K;

              double effusionMaxPower_W = EFF_DEFAULT_MAX_W;
              if (std::isfinite(rj.requested_heater_cap_W) &&
                  rj.requested_heater_cap_W > 0.0) {
                effusionMaxPower_W = rj.requested_heater_cap_W;
              }

              // Source demand is phase-aware.
              //
              // Old behavior:
              // - only beam-enabled growth rows got a meaningful source target
              // - all beam-off phases effectively idled the source
              //
              // New behavior:
              // - any phase with source_control_on may request positive heating
              // - SOURCE_DEGAS and SOAK can therefore heat and hold the source
              //   with mbe_flag still forced off
              // - heater_W remains a cap, not a setpoint
              if (!phaseIntent.source_control_on ||
                  target_T_K <= (DEFAULT_IDLE_EFFUSION_TARGET_K + 10.0)) {
                effusionDemand_W = 0.0;
              } else {
                const double p_ff_W =
                    EFF_H_W_PER_K * std::max(0.0, target_T_K - EFF_T_ENV_K);

                const double p_p_W =
                    (eff_temp_error_K > 0.0)
                        ? (EFF_KP_W_PER_K * eff_temp_error_K)
                        : 0.0;

                effusionDemand_W =
                    std::clamp(p_ff_W + p_p_W, 0.0, effusionMaxPower_W);
              }

              if (!std::isfinite(effusionDemand_W) || effusionDemand_W < 0.0) {
                effusionDemand_W = 0.0;
              }
              
              substrateDemand_W = substrateHeater.computePowerRequestW();

              const auto effBand   = effCell.getThermalBandState(0.90, 1.05);
              const auto waferBand = substrateHeater.getThermalBandState();

              const bool effBelow    = (effBand == EffusionCell::ThermalBandState::BelowTargetBand);
              const bool effWithin   = (effBand == EffusionCell::ThermalBandState::WithinTargetBand ||
                                        effBand == EffusionCell::ThermalBandState::Idle);
              const bool effAbove    = (effBand == EffusionCell::ThermalBandState::AboveTargetBand);

              const bool waferBelow  = (waferBand == SubstrateHeater::ThermalBandState::BelowTargetBand);
              const bool waferWithin = (waferBand == SubstrateHeater::ThermalBandState::WithinTargetBand ||
                                        waferBand == SubstrateHeater::ThermalBandState::Idle);
              const bool waferAbove  = (waferBand == SubstrateHeater::ThermalBandState::AboveTargetBand);

              wafer_ready = waferWithin;
              effusion_ready = effWithin;

              const JobRunState oldState = rj.state;

              // A phase becomes executable only when the subsystems that matter
              // for that phase are in-band.
              //
              // Important:
              // - substrate readiness only matters when substrate control is on
              // - source readiness matters for phases that actively depend on
              //   source conditioning, including SOURCE_DEGAS and SOAK
              // - beam permission is still separate from source readiness
              const bool substrate_ready_for_phase =
                  phaseIntent.substrate_control_on ? waferWithin : true;

              const bool effusion_ready_for_phase =
                  phaseIntent.source_ready_required ? effWithin : true;

              const bool phase_ready_for_execution =
                  substrate_ready_for_phase && effusion_ready_for_phase;

              phase_ready_for_execution_log =
                  phase_ready_for_execution ? 1.0 : 0.0;

              const bool growth_execution_row =
                  deposition_requested && phaseIntent.growth_like_phase;

              const bool execution_hold_lost_after_live_start =
                  growth_execution_row &&
                  rj.has_started_live_execution &&
                  !phase_ready_for_execution;

              if (execution_hold_lost_after_live_start) {
                const bool first_hold_fault = !rj.live_execution_hold_faulted;
                // Important semantic rule:
                // once live growth execution has begun, loss of required thermal hold
                // is no longer treated as harmless prep. Stay in live-execution semantics
                // so failure accounting can continue toward abort.
                rj.state = JobRunState::LiveDeposition;
                rj.live_execution_hold_faulted = true;

                  if (first_hold_fault) {
                    log_msg("[sched] first hold fault detected...\n");
                  }

                live_active = 1.0;
                prep_mode_code = 0.0;

              } else if ((phaseIntent.substrate_control_on && waferBelow) ||
                        (phaseIntent.source_ready_required && effBelow)) {

                rj.state = JobRunState::Warming;
                warmup_active = 1.0;
                prep_mode_code = 1.0;

                if (rj.actual_thermal_prep_start_tick < 0) {
                  rj.actual_thermal_prep_start_tick = tickIndex;
                }
                if (rj.actual_warmup_start_tick < 0) {
                  rj.actual_warmup_start_tick = tickIndex;
                }

              } else if ((phaseIntent.substrate_control_on && waferAbove) ||
                        (phaseIntent.source_ready_required && effAbove)) {

                rj.state = JobRunState::Cooling;
                cooldown_active = 1.0;
                prep_mode_code = 2.0;

                if (rj.actual_thermal_prep_start_tick < 0) {
                  rj.actual_thermal_prep_start_tick = tickIndex;
                }
                if (rj.actual_cooldown_start_tick < 0) {
                  rj.actual_cooldown_start_tick = tickIndex;
                }

              } else if (!phase_ready_for_execution) {
                rj.state = JobRunState::ThermalPrep;
                thermal_prep_active = 1.0;
                prep_mode_code = 3.0;

                if (rj.actual_thermal_prep_start_tick < 0) {
                  rj.actual_thermal_prep_start_tick = tickIndex;
                }

              } else {
                // Ready beam-off timed phases execute in ThermalPrep.
                // Ready growth-like phases execute in LiveDeposition.
                if (deposition_requested) {
                  rj.state = JobRunState::LiveDeposition;
                  live_active = 1.0;
                  prep_mode_code = 0.0;

                  if (rj.actual_phase_start_tick < 0) {
                    rj.actual_phase_start_tick = tickIndex;
                  }
                  if (rj.actual_deposition_start_tick < 0) {
                    rj.actual_deposition_start_tick = tickIndex;
                  }

                  rj.has_started_live_execution = true;
                  rj.live_execution_hold_faulted = false;

                } else {
                  rj.state = JobRunState::ThermalPrep;
                  thermal_prep_active = 1.0;
                  prep_mode_code = 0.0;

                  if (rj.actual_phase_start_tick < 0) {
                    rj.actual_phase_start_tick = tickIndex;
                  }
                }
              }

              const bool substrate_execution_monitor_active =
                  (rj.state == JobRunState::LiveDeposition) ||
                  (rj.state == JobRunState::ThermalPrep && !deposition_requested && phase_ready_for_execution);

              substrateHeater.setFailureMonitorArmed(
                  substrate_execution_monitor_active);

              if (rj.state != oldState) {
                std::ostringstream oss;
                oss << "[sched] tick=" << tickIndex
                    << " job " << controllingJobIndex
                    << " state " << jobRunStateName(oldState)
                    << " -> " << jobRunStateName(rj.state)
                    << " (remaining_phase_ticks=" << rj.remaining_phase_ticks
                    << ", remaining_live_ticks=" << rj.remaining_live_ticks
                    << ", eff_T=" << effCell.getTemperatureK()
                    << ", eff_target=" << effCell.getTargetTempK()
                    << ", sub_T=" << substrateHeater.substrateTempK()
                    << ", sub_target=" << substrateHeater.targetTempK()
                    << ")\n";
                log_msg(oss.str());
              }

              // The physical beam is only active during true live deposition.
              //
              // This preserves the key scheduler invariant:
              // - source heating may be active in beam-off phases
              // - substrate heating may be active in beam-off phases
              // - actual deposition still occurs only in LiveDeposition
              // - SPARTA still sees a legal floor flux when the beam is off
                const bool beam_may_fire_now =
                    (rj.state == JobRunState::LiveDeposition) &&
                    deposition_requested &&
                    !rj.live_execution_hold_faulted &&
                    phase_ready_for_execution;

                mbe_flag = beam_may_fire_now ? 1.0 : 0.0;
                growth_flux_cm2s = (mbe_flag > 0.5) ? raw_job_flux_cm2s : 0.0;

              // Delay is measured to the realized start of execution for the row.
              // For growth-like phases this is usually the actual deposition start.
              // For beam-off timed phases this is the actual phase start.
              if (rj.actual_phase_start_tick >= 0) {
                delay_from_requested_start_log =
                    static_cast<double>(rj.actual_phase_start_tick - rj.requested_start_tick);
              } else if (rj.actual_deposition_start_tick >= 0) {
                delay_from_requested_start_log =
                    static_cast<double>(rj.actual_deposition_start_tick - rj.requested_start_tick);
              } else {
                delay_from_requested_start_log =
                    static_cast<double>(tickIndex - rj.requested_start_tick);
              }

              queued_active       = (rj.state == JobRunState::Queued) ? 1.0 : 0.0;
              warmup_active       = (rj.state == JobRunState::Warming) ? 1.0 : warmup_active;
              cooldown_active     = (rj.state == JobRunState::Cooling) ? 1.0 : cooldown_active;
              thermal_prep_active = (rj.state == JobRunState::ThermalPrep) ? 1.0 : thermal_prep_active;
              live_active         = (rj.state == JobRunState::LiveDeposition) ? 1.0 : live_active;
              done_active         = (rj.state == JobRunState::Done) ? 1.0 : done_active;
              aborted_active      = (rj.state == JobRunState::Aborted) ? 1.0 : aborted_active;

              scheduler_state_code_log = jobRunStateCode(rj.state);
            }


          } else if (!jobs.empty()) {
            raw_job_flux_cm2s    = 0.0;
            deposition_requested = false;

            if (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0) {
              sparta_flux_cm2s = FWAFFER_FLOOR_CM2S;
            } else {
              sparta_flux_cm2s = last_Fwafer_sent;
            }

            mbe_flag   = 0.0;
            target_T_K = DEFAULT_IDLE_EFFUSION_TARGET_K;

            scheduler_substrate_target_K_log = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
            effCell.setTargetTempK(target_T_K);

            effusionDemand_W = 0.0;
            // No row currently owns thermal control.
            // Until SubstrateHeater is refactored for explicit recipe targets,
            // this falls back to the legacy idle behavior.
            substrateHeater.setJobState(
                -1,
                /*job_active=*/false,
                0.0,
                false,
                DEFAULT_IDLE_SUBSTRATE_TARGET_K
            );

            substrateDemand_W = substrateHeater.computePowerRequestW();
            wafer_ready = true;
            effusion_ready = true;

            scheduler_state_code_log = jobRunStateCode(JobRunState::Pending);
          } else {
            raw_job_flux_cm2s    = 0.0;
            deposition_requested = false;

            if (std::isnan(last_Fwafer_sent) || last_Fwafer_sent <= 0.0) {
              sparta_flux_cm2s = FWAFFER_FLOOR_CM2S;
            } else {
              sparta_flux_cm2s = last_Fwafer_sent;
            }

            mbe_flag   = 0.0;
            target_T_K = DEFAULT_IDLE_EFFUSION_TARGET_K;
            scheduler_substrate_target_K_log = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
            effCell.setTargetTempK(target_T_K);

            // changed so that 
            effusionDemand_W = 0.0;

            substrateHeater.setJobState(
                -1,
                /*job_active=*/false,
                0.0,
                false,
                DEFAULT_IDLE_SUBSTRATE_TARGET_K
            );

            substrateDemand_W = substrateHeater.computePowerRequestW();
            wafer_ready = true;
            effusion_ready = true;

            scheduler_state_code_log = jobRunStateCode(JobRunState::Pending);
          }

          growth.setBeamState(jobIndexForGrowth,
                              mbe_flag > 0.5,
                              growth_flux_cm2s);

                              // earlier logging was here, but moved after decision logic for cleaner logs and to capture the effect of decisions on SPARTA state in the same tick

          // ---- 3) Push Fwafer + mbe_active into params.inc when needed ----
          if (std::isfinite(sparta_flux_cm2s) && sparta_flux_cm2s > 0.0) {
            bool need_update = false;

            if (std::isnan(last_Fwafer_sent) ||
                sparta_flux_cm2s != last_Fwafer_sent) {
              need_update = true;
            }
            if (std::isnan(last_mbe_sent) ||
                mbe_flag != last_mbe_sent) {
              need_update = true;
            }

            if (need_update) {
              std::ostringstream oss;
              oss << "[job] tick=" << tickIndex
                  << " update params.inc: raw_job_flux_cm2s=" << raw_job_flux_cm2s
                  << ", sparta_flux_cm2s=" << sparta_flux_cm2s
                  << ", deposition_requested=" << (deposition_requested ? 1 : 0)
                  << ", mbe_active=" << mbe_flag << "\n";
              log_msg(oss.str());

              log_rank_progress(tickIndex, "before-write_params_inc");
              write_params_inc(sparta_flux_cm2s, mbe_flag, rank, args.inputDir, log_msg);
              log_rank_progress(tickIndex, "after-write_params_inc");

              log_rank_progress(tickIndex, "before-markDirtyReload");
              wake.markDirtyReload();
              log_rank_progress(tickIndex, "after-markDirtyReload");

              last_Fwafer_sent = sparta_flux_cm2s;
              last_mbe_sent    = mbe_flag;
            }
          }

          // ---- 4) Set heater demand (only log when it changes) ----
          double totalHeaterDemand_W = effusionDemand_W + substrateDemand_W;

          if (std::isnan(last_heater_set) || totalHeaterDemand_W != last_heater_set) {
            std::ostringstream oss;
            oss << "[job] tick=" << tickIndex
                << " control_state:"
                << " raw_job_flux_cm2s=" << raw_job_flux_cm2s
                << ", sparta_flux_cm2s=" << sparta_flux_cm2s
                << ", deposition_requested=" << (deposition_requested ? 1 : 0)
                << ", effusionDemand_W=" << effusionDemand_W
                << ", substrateDemand_W=" << substrateDemand_W
                << ", totalHeaterDemand_W=" << totalHeaterDemand_W
                << ", wafer_ready=" << (wafer_ready ? 1 : 0)
                << ", effusion_ready=" << (effusion_ready ? 1 : 0)
                << ", mbe_flag=" << mbe_flag
                << "\n";
            log_msg(oss.str());
            last_heater_set = totalHeaterDemand_W;
          }

          heater.setEffusionDemand(effusionDemand_W);
          heater.setSubstrateDemand(substrateDemand_W);

          // While wafer is not ready, prioritize wafer heating.
          // (Once ready, HeaterBank can share or go proportional.)
          heater.setPrioritySubstrate(!wafer_ready);


          // ---- 5) Tick harness + WakeChamber ----
          std::ostringstream oss;
          oss << "[wake] tick=" << tickIndex
              << " t=" << t_phys
              << " s : BEFORE engine.tick() + wake.tick()\n";
          log_msg(oss.str());

          // C++ harness first, then SPARTA
          log_rank_progress(tickIndex, "before-engine-tick");
          engine.tick();
          log_rank_progress(tickIndex, "after-engine-tick");

          TickContext ctx{ tickIndex, t_phys, dt };
          log_rank_progress(tickIndex, "before-wake-tick");
          wake.tick(ctx);
          log_rank_progress(tickIndex, "after-wake-tick");

          std::ostringstream oss2;
          oss2 << "[wake] tick=" << tickIndex
               << " t=" << t_phys
               << " s : AFTER engine.tick() + wake.tick()\n";
          log_msg(oss2.str());

          // ---- 6) After ticking, evaluate job health and consume execution time
          if (controllingJobIndex >= 0) {
            RuntimeJobState& rj = runtimeJobs[controllingJobIndex];

            const bool live_execution_active =
              (rj.state == JobRunState::LiveDeposition &&
              deposition_requested &&
              rj.has_started_live_execution);

            const bool nongrowth_execution_active =
                (rj.state == JobRunState::ThermalPrep &&
                 !deposition_requested &&
                 phase_ready_for_execution_log > 0.5);

            bool wafer_gate_fail = substrateHeater.jobFailed();


            if (live_execution_active) {
              const bool execution_hold_fault_active = rj.live_execution_hold_faulted;
              if (execution_hold_fault_active) {
                std::ostringstream oss;
                oss << "[sched] tick=" << tickIndex
                    << " job " << controllingJobIndex
                    << " execution-time thermal hold lost"
                    << " (eff_ready=" << (effusion_ready ? 1 : 0)
                    << ", wafer_ready=" << (wafer_ready ? 1 : 0)
                    << ", mbe_flag=" << mbe_flag
                    << ")\n";
                log_msg(oss.str());
              }
              double P_actual = effCell.getLastHeatInputW();

              // Update RC temp proxy for the active live-deposition interval.
              {
                const double C_J_PER_K = 800.0;
                const double H_W_PER_K = 0.8;
                const double T_ENV_K   = 300.0;

                double net_W = P_actual - H_W_PER_K * (temp_proxy_K - T_ENV_K);
                double dT    = (net_W / C_J_PER_K) * dt;
                temp_proxy_K += dT;

                if (!std::isfinite(temp_proxy_K)) temp_proxy_K = T_ENV_K;
                if (temp_proxy_K < 0.0)           temp_proxy_K = 0.0;
              }

              // Under-flux gate while live
              double flux_ratio = 1.0;
              if (effusionDemand_W > 0.0) {
                flux_ratio = P_actual / effusionDemand_W;
              }
              if (!std::isfinite(flux_ratio)) {
                flux_ratio = 0.0;
              }

              if (flux_ratio < MIN_FLUX_FRACTION) {
                underflux_streak += 1;
              } else {
                underflux_streak = 0;
              }

              bool flux_gate_fail = (underflux_streak >= UNDERFLUX_LIMIT_TICKS);

              // Temperature gate while live
              bool temp_gate_fail = false;

              double actual_effusion_temp_K = effCell.getTemperatureK();
              double temp_ratio = (target_T_K > 0.0) ? (actual_effusion_temp_K / target_T_K) : 0.0;

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

              g_underflux_streak_for_log = underflux_streak;
              g_temp_miss_streak_for_log = temp_miss_streak;

              // Abort only during true live deposition
              if (flux_gate_fail || temp_gate_fail || wafer_gate_fail) {
                rj.aborted = true;
                rj.state   = JobRunState::Aborted;

                std::ostringstream joss;
                joss << "[job] tick=" << tickIndex
                     << " ABORTING job index " << controllingJobIndex
                     << " due to ";

                bool any = false;
                if (flux_gate_fail)  { joss << (any ? " + " : "") << "under-flux";       any = true; }
                if (temp_gate_fail)  { joss << (any ? " + " : "") << "temperature-miss"; any = true; }
                if (wafer_gate_fail) { joss << (any ? " + " : "") << "wafer-temp-miss";  any = true; }

                joss << " (underflux_streak=" << underflux_streak
                     << ", temp_miss_streak=" << temp_miss_streak
                     << ", eff_temp_K=" << effCell.getTemperatureK()
                     << ", target_T_K=" << target_T_K
                     << ", flux_ratio=" << flux_ratio
                     << ", T_sub_K=" << substrateHeater.substrateTempK()
                     << ", T_sub_target_K=" << substrateHeater.targetTempK()
                     << ", sub_temp_miss_streak=" << substrateHeater.tempMissStreak()
                     << ")\n";
                log_msg(joss.str());

                growth.markJobAborted(controllingJobIndex);
                engine.markJobFailedThisTick();

                double F_for_abort = last_Fwafer_sent;
                if (!std::isfinite(F_for_abort) || F_for_abort <= 0.0) {
                  F_for_abort = FWAFFER_FLOOR_CM2S;
                }

                log_rank_progress(tickIndex, "before-abort-write_params_inc");
                write_params_inc(F_for_abort, 0.0, rank, args.inputDir, log_msg);
                log_rank_progress(tickIndex, "after-abort-write_params_inc");

                log_rank_progress(tickIndex, "before-abort-markDirtyReload");
                wake.markDirtyReload();
                log_rank_progress(tickIndex, "after-abort-markDirtyReload");

                last_Fwafer_sent = F_for_abort;
                last_mbe_sent    = 0.0;
                logJobIndexAfterAccounting = controllingJobIndex;
                rj.has_started_live_execution = false;
                rj.live_execution_hold_faulted = false;

                controllingJobIndex = -1;

                underflux_streak = 0;
                temp_miss_streak = 0;
                g_underflux_streak_for_log = 0;
                g_temp_miss_streak_for_log = 0;
                temp_proxy_K = 300.0;


                last_heater_set = std::numeric_limits<double>::quiet_NaN();
                last_effusion_set = std::numeric_limits<double>::quiet_NaN();
                last_substrate_set = std::numeric_limits<double>::quiet_NaN();
              } else {
                // Growth-like phases consume both phase time and live deposition time.
                rj.phase_ticks_completed += 1;
                rj.remaining_phase_ticks -= 1;

                rj.live_ticks_completed += 1;
                rj.remaining_live_ticks -= 1;

                std::ostringstream oss;
                oss << "[sched] tick=" << tickIndex
                    << " job " << controllingJobIndex
                    << " completed one live deposition tick"
                    << " (phase_completed=" << rj.phase_ticks_completed
                    << ", phase_remaining=" << rj.remaining_phase_ticks
                    << ", live_completed=" << rj.live_ticks_completed
                    << ", live_remaining=" << rj.remaining_live_ticks
                    << ")\n";
                log_msg(oss.str());

                if (rj.remaining_phase_ticks <= 0) {
                  rj.done = true;
                  rj.state = JobRunState::Done;
                  rj.actual_phase_end_tick = tickIndex;

                  if (rj.actual_deposition_start_tick >= 0 && rj.actual_deposition_end_tick < 0) {
                    rj.actual_deposition_end_tick = tickIndex;
                  }

                  std::ostringstream done_oss;
                  done_oss << "[sched] tick=" << tickIndex
                           << " job " << controllingJobIndex
                           << " DONE"
                           << " (requested_start=" << rj.requested_start_tick
                           << ", actual_phase_start=" << rj.actual_phase_start_tick
                           << ", actual_phase_end=" << rj.actual_phase_end_tick
                           << ", phase_ticks_completed=" << rj.phase_ticks_completed
                           << ", live_ticks_completed=" << rj.live_ticks_completed
                           << ")\n";
                  log_msg(done_oss.str());
                  logJobIndexAfterAccounting = controllingJobIndex;

                  rj.has_started_live_execution = false;
                  rj.live_execution_hold_faulted = false;

                  controllingJobIndex = -1;
                  underflux_streak = 0;
                  temp_miss_streak = 0;
                  g_underflux_streak_for_log = 0;
                  g_temp_miss_streak_for_log = 0;
                  temp_proxy_K = 300.0;
                  last_heater_set = std::numeric_limits<double>::quiet_NaN();
                  last_effusion_set = std::numeric_limits<double>::quiet_NaN();
                  last_substrate_set = std::numeric_limits<double>::quiet_NaN();
                }
              }
            } else if (nongrowth_execution_active) {
              if (wafer_gate_fail) {
                rj.aborted = true;
                rj.state   = JobRunState::Aborted;

                std::ostringstream joss;
                joss << "[job] tick=" << tickIndex
                     << " ABORTING non-growth job index " << controllingJobIndex
                     << " due to wafer-temp-miss"
                     << " (T_sub_K=" << substrateHeater.substrateTempK()
                     << ", T_sub_target_K=" << substrateHeater.targetTempK()
                     << ", sub_temp_miss_streak=" << substrateHeater.tempMissStreak()
                     << ")\n";
                log_msg(joss.str());

                growth.markJobAborted(controllingJobIndex);
                engine.markJobFailedThisTick();

                double F_for_abort = last_Fwafer_sent;
                if (!std::isfinite(F_for_abort) || F_for_abort <= 0.0) {
                  F_for_abort = FWAFFER_FLOOR_CM2S;
                }

                log_rank_progress(tickIndex, "before-abort-write_params_inc");
                write_params_inc(F_for_abort, 0.0, rank, args.inputDir, log_msg);
                log_rank_progress(tickIndex, "after-abort-write_params_inc");

                log_rank_progress(tickIndex, "before-abort-markDirtyReload");
                wake.markDirtyReload();
                log_rank_progress(tickIndex, "after-abort-markDirtyReload");

                last_Fwafer_sent = F_for_abort;
                last_mbe_sent    = 0.0;
                logJobIndexAfterAccounting = controllingJobIndex;

                rj.has_started_live_execution = false;
                rj.live_execution_hold_faulted = false;

                controllingJobIndex = -1;
                underflux_streak = 0;
                temp_miss_streak = 0;
                g_underflux_streak_for_log = 0;
                g_temp_miss_streak_for_log = 0;
                temp_proxy_K = 300.0;
                last_heater_set = std::numeric_limits<double>::quiet_NaN();
                last_effusion_set = std::numeric_limits<double>::quiet_NaN();
                last_substrate_set = std::numeric_limits<double>::quiet_NaN();

              } else {
                // Beam-off timed phases consume phase time only.
                underflux_streak = 0;
                temp_miss_streak = 0;
                g_underflux_streak_for_log = 0;
                g_temp_miss_streak_for_log = 0;

                rj.phase_ticks_completed += 1;
                rj.remaining_phase_ticks -= 1;

                std::ostringstream oss;
                oss << "[sched] tick=" << tickIndex
                    << " job " << controllingJobIndex
                    << " completed one non-growth phase tick"
                    << " (phase_completed=" << rj.phase_ticks_completed
                    << ", phase_remaining=" << rj.remaining_phase_ticks
                    << ")\n";
                log_msg(oss.str());

                if (rj.remaining_phase_ticks <= 0) {
                  rj.done = true;
                  rj.state = JobRunState::Done;
                  rj.actual_phase_end_tick = tickIndex;

                  std::ostringstream done_oss;
                  done_oss << "[sched] tick=" << tickIndex
                           << " job " << controllingJobIndex
                           << " DONE"
                           << " (requested_start=" << rj.requested_start_tick
                           << ", actual_phase_start=" << rj.actual_phase_start_tick
                           << ", actual_phase_end=" << rj.actual_phase_end_tick
                           << ", phase_ticks_completed=" << rj.phase_ticks_completed
                           << ", live_ticks_completed=" << rj.live_ticks_completed
                           << ")\n";
                  log_msg(done_oss.str());

                  logJobIndexAfterAccounting = controllingJobIndex;
                  controllingJobIndex = -1;
                  last_heater_set = std::numeric_limits<double>::quiet_NaN();
                  last_effusion_set = std::numeric_limits<double>::quiet_NaN();
                  last_substrate_set = std::numeric_limits<double>::quiet_NaN();
                }
              }

            } else {
              // Queue, warmup, cooldown, and not-yet-ready thermal-prep ticks
              // do not consume execution time and do not accumulate live streaks.
              underflux_streak = 0;
              temp_miss_streak = 0;
              g_underflux_streak_for_log = 0;
              g_temp_miss_streak_for_log = 0;
            }

          } else {
            // No controller -> clear streaks in logs.
            underflux_streak = 0;
            temp_miss_streak = 0;
            g_underflux_streak_for_log = 0;
            g_temp_miss_streak_for_log = 0;
          }

          // ---- 7) Emit post-accounting logs for the same tick ----
          emitPostAccountingLogs(
              tickIndex,
              t_phys,
              logJobIndexAfterAccounting,
              raw_job_flux_cm2s,
              sparta_flux_cm2s,
              deposition_requested,
              effusionDemand_W,
              substrateDemand_W,
              target_T_K,
              scheduler_substrate_target_K_log,
              phase_ready_for_execution_log,
              mbe_flag,
              effusion_ready,
              wafer_ready);

        } // end if (isLeader)

        // ---------------- SPARTA coupling block ----------------
        if (i % args.coupleEvery == 0) {
          log_rank_progress(tickIndex, "before-runIfDirtyOrAdvanceCollective");

          if (isLeader) {
            std::ostringstream oss;
            oss << "[cpl] tick=" << tickIndex
                << " ENTER wake.runIfDirtyOrAdvanceCollective(spartaBlock="
                << args.spartaBlock << ")\n";
            log_msg(oss.str());
          }

          wake.runIfDirtyOrAdvanceCollective(args.spartaBlock);

          if (isLeader) {
            std::ostringstream oss;
            oss << "[cpl] tick=" << tickIndex
                << " EXIT  wake.runIfDirtyOrAdvanceCollective(...)\n";
            log_msg(oss.str());
          }

          log_rank_progress(tickIndex, "after-runIfDirtyOrAdvanceCollective");
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
