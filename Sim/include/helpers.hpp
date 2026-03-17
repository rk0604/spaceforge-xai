#pragma once

#include <string>
#include <vector>
#include <functional>

namespace SimHelpers {

// Logger function type used by helpers.
using LogFn = std::function<void(const std::string&)>;

// -----------------------------------------------------------------------------
// CLI arguments / configuration
// -----------------------------------------------------------------------------
// These values control simulator startup and top-level execution behavior.
// They are intentionally lightweight so parsing can stay dependency-free.
struct Args {
  std::string mode     = "dual";            // "dual", "legacy", "wake", or "power"
  std::string wakeDeck = "in.wake_harness";
  std::string effDeck  = "in.effusion";     // kept for compatibility, currently unused
  std::string inputDir = "input";

  int  nWake       = -1;    // kept for compatibility, currently unused
  int  coupleEvery = 10;    // advance SPARTA every X engine ticks
  int  spartaBlock = 200;   // run N SPARTA steps per advance
  bool showHelp    = false;

  int    nticks = 500;      // engine ticks to run
  double dt     = 60.0;     // seconds per engine tick
};

// -----------------------------------------------------------------------------
// Recipe phase semantics
// -----------------------------------------------------------------------------
// PhaseCode makes job rows recipe-aware without pretending the simulator is
// a chemistry-complete digital twin. The scheduler can use these codes to
// interpret beam permission, substrate control behavior, and time accounting.
enum class PhaseCode {
  IDLE = 0,
  SOURCE_DEGAS,
  OXIDE_DESORB,
  SOAK,
  NUCLEATE,
  GROWTH,
  ANNEAL,
  COOLDOWN
};

// Convert a phase code to a stable string for logs and CSV output.
const char* phaseCodeName(PhaseCode code);

// Parse a phase code from text. Throws std::runtime_error on invalid input.
PhaseCode parsePhaseCode(const std::string& text);

// -----------------------------------------------------------------------------
// Job schedule row
// -----------------------------------------------------------------------------
// This struct supports both the legacy 4-column schedule and the newer
// recipe-aware 8-column schedule.
//
// Legacy format:
//   start_tick end_tick wafer_flux_cm2s heater_W
//
// Recipe-aware format:
//   start_tick end_tick wafer_flux_cm2s heater_cap_W
//   mbe_on substrate_on phase_code substrate_target_K
//
// Semantics:
// - Fwafer_cm2s is the requested deposition flux when beam-on is permitted.
// - heater_W is an effusion heater cap, not a direct source temperature setpoint.
// - mbe_on controls whether beam-on deposition is allowed for the row.
// - substrate_on controls whether substrate temperature control is active.
// - phase_code gives explicit process semantics.
// - substrate_target_K is the requested substrate target temperature for the row.
struct Job {
  int    start_tick          = 0;      // inclusive release bound
  int    end_tick            = -1;     // exclusive or inclusive by caller policy
  double Fwafer_cm2s         = 0.0;    // requested wafer flux
  double heater_W            = 0.0;    // effusion heater cap in watts

  int    mbe_on              = 0;      // 0 or 1
  int    substrate_on        = 0;      // 0 or 1
  PhaseCode phase_code       = PhaseCode::IDLE;
  double substrate_target_K  = 300.0;  // explicit substrate target for the row
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
// Global floor so Fwafer_cm2s never goes fully to zero in SPARTA mixture logic.
// This protects the wake-side mixture setup while beam gating remains explicit.
extern const double FWAFFER_FLOOR_CM2S;

// Default idle-like substrate temperature used by helper fallback logic.
extern const double DEFAULT_IDLE_SUBSTRATE_TARGET_K;

// -----------------------------------------------------------------------------
// CLI helpers
// -----------------------------------------------------------------------------
Args parse_args(int argc, char** argv);
void print_usage();

// -----------------------------------------------------------------------------
// Physics helper
// -----------------------------------------------------------------------------
// Map requested wafer flux into a heater power request or cap estimate.
// This helper is intentionally simple and remains an abstraction layer.
double fluxToHeaterPower(double Fwafer_cm2s);

// -----------------------------------------------------------------------------
// Job parsing and validation helpers
// -----------------------------------------------------------------------------
// Parse a single schedule line.
//
// Returns true if parsing succeeds.
// Returns false if parsing fails and writes a human-readable reason into err.
//
// Supported formats:
// - 8-column recipe-aware format
// - 4-column legacy format
//
// Legacy rows are upgraded with defaults:
// - positive-flux rows become GROWTH rows with beam and substrate enabled
// - non-positive-flux rows become IDLE rows with beam and substrate disabled
bool parseJobLine(const std::string& line, Job& job, std::string& err);

// Validate semantic consistency of a parsed job row.
// Returns true if valid. Returns false and fills err if invalid.
bool validateJob(const Job& job, std::string& err);

// -----------------------------------------------------------------------------
// Duration helpers
// -----------------------------------------------------------------------------
// Phase duration is the total scheduled recipe time represented by the row.
// This is independent of whether beam-on deposition occurs.
int derivePhaseDurationTicks(const Job& job);

// Live deposition duration is the beam-on growth time owed by the row.
// This is only nonzero for growth-like rows with positive flux and beam enabled.
int deriveLiveDepositionTicks(const Job& job);

// True for rows that represent beam-on growth execution.
bool isGrowthPhase(PhaseCode code);

// True for rows that represent timed thermal processing without beam-on growth.
bool isNonGrowthTimedPhase(PhaseCode code);

// -----------------------------------------------------------------------------
// params.inc writer
// -----------------------------------------------------------------------------
// Rewrite inputDir/params.inc with the latest wake parameters.
//
// MPI behavior:
// - Only rank 0 writes the file.
// - This function performs no MPI collectives and no barrier.
// - Synchronization is handled by the caller.
void write_params_inc(double Fwafer_cm2s,
                      double mbe_active,
                      int rank,
                      const std::string& inputDir,
                      LogFn log_fn);

} // namespace SimHelpers