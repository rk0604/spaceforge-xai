#pragma once

#include <functional>
#include <string>
#include <vector>

namespace SimHelpers {

// Logger callback used by helper utilities that need to report file I/O or
// parsing issues back to the caller without owning any global logging state.
using LogFn = std::function<void(const std::string&)>;

// -----------------------------------------------------------------------------
// CLI arguments / configuration
// -----------------------------------------------------------------------------
// These values control simulator startup and top-level execution behavior.
// They are intentionally lightweight so parsing can stay dependency-free.
struct Args {
  std::string mode     = "dual";
  std::string wakeDeck = "in.wake_harness";
  std::string effDeck  = "in.effusion";
  std::string inputDir = "input";

  int  nWake       = -1;
  int  coupleEvery = 10;
  int  spartaBlock = 200;
  bool showHelp    = false;

  int    nticks = 500;
  double dt     = 60.0;
};

// -----------------------------------------------------------------------------
// Recipe phase semantics
// -----------------------------------------------------------------------------
// PhaseCode gives each schedule row an explicit process meaning. These codes are
// not intended to turn the simulator into a chemistry-complete MBE digital
// twin. They give the scheduler enough structure to distinguish beam-on growth
// from beam-off thermal preparation and cooldown behavior.
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

// Convert a phase code to a stable text label for logs and CSV output.
const char* phaseCodeName(PhaseCode code);

// Parse a phase code from text. Throws std::runtime_error on invalid input.
PhaseCode parsePhaseCode(const std::string& text);

// True for phases that represent beam-on deposition execution.
bool isGrowthPhase(PhaseCode code);

// True for phases that represent timed processing with no beam-on growth.
bool isNonGrowthTimedPhase(PhaseCode code);

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
// - Fwafer_cm2s is the requested wafer flux when beam-on deposition is allowed.
// - heater_W is an effusion heater cap, not a direct source temperature setpoint.
// - mbe_on controls beam permission, not source heating permission.
// - substrate_on controls whether substrate thermal control is active.
// - phase_code gives the scheduler explicit phase semantics.
// - substrate_target_K is the row-level substrate temperature target.
struct Job {
  int    start_tick         = 0;
  int    end_tick           = -1;
  double Fwafer_cm2s        = 0.0;
  double heater_W           = 0.0;

  int    mbe_on             = 0;
  int    substrate_on       = 0;
  PhaseCode phase_code      = PhaseCode::IDLE;
  double substrate_target_K = 300.0;
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
// Global floor so SPARTA mixture variables never go to an illegal zero value.
// Physical deposition remains controlled by mbe_active and scheduler logic.
extern const double FWAFFER_FLOOR_CM2S;

// Default ambient-like substrate target used when no recipe row owns control.
extern const double DEFAULT_IDLE_SUBSTRATE_TARGET_K;

// Default ambient-like source target used when no recipe row owns control.
extern const double DEFAULT_IDLE_EFFUSION_TARGET_K;

// -----------------------------------------------------------------------------
// CLI helpers
// -----------------------------------------------------------------------------
Args parse_args(int argc, char** argv);
void print_usage();

// -----------------------------------------------------------------------------
// Generic source-side thermal mapping helpers
// -----------------------------------------------------------------------------
// Map requested wafer flux to a notional source target temperature in Kelvin.
// This remains a reduced-order scheduler-facing abstraction. It is monotonic in
// flux and intended for thermal targeting, logging, and readiness gating rather
// than species-specific vapor-pressure prediction.
double targetTempForFlux(double Fwafer_cm2s);

// Map a reference growth flux to a hotter beam-off source degas target.
// This encodes the idea that source degassing is hotter than idle and is often
// performed at or above nominal operating conditions, while still keeping the
// model computationally simple.
double sourceDegasTargetTempForFlux(double reference_flux_cm2s);

// Map a reference growth flux to a near-growth beam-off soak target.
// This supports recipe phases that are intended to hold the source near an
// upcoming growth operating point without opening the beam.
double sourceSoakTargetTempForFlux(double reference_flux_cm2s);

// Map a reference growth flux to a reduced-demand source anneal target.
// This is lower than full growth operation but still allows controlled source
// thermal relaxation when the process semantics call for it.
double sourceAnnealTargetTempForFlux(double reference_flux_cm2s);

// Existing reduced-order flux to heater power abstraction used elsewhere in the
// simulator. This is kept for compatibility with current warmup diagnostics.
double fluxToHeaterPower(double Fwafer_cm2s);

// -----------------------------------------------------------------------------
// Phase-aware source policy
// -----------------------------------------------------------------------------
// SourceTargetMode makes logging and scheduler policy inspection easier. It lets
// the caller distinguish why a target was chosen without embedding that meaning
// into EffusionCell itself.
enum class SourceTargetMode {
  Idle = 0,
  DegasHot,
  StandbyNearGrowth,
  GrowthFluxDerived,
  AnnealReduced,
  PassiveCooldown
};

// Convert a source target mode to a stable text label for logs and CSV output.
const char* sourceTargetModeName(SourceTargetMode mode);

// This struct describes scheduler-facing source intent for a single recipe row.
// It does not represent the physical source state itself. EffusionCell should
// still own thermal state evolution and actual temperature dynamics.
struct SourcePhasePolicy {
  bool source_control_on       = false;
  bool source_ready_required   = false;
  bool beam_allowed            = false;
  bool growth_like_execution   = false;
  bool flux_influences_target  = false;
  bool source_may_idle         = true;

  SourceTargetMode target_mode = SourceTargetMode::Idle;
  double effusion_target_K     = DEFAULT_IDLE_EFFUSION_TARGET_K;
};

// Resolve which flux should anchor the source target for a given phase.
// For growth-like phases this is typically the current row flux.
// For beam-off pre-growth phases this may fall back to the next growth flux.
double resolveReferenceFluxForSourcePhase(PhaseCode phase_code,
                                          double row_flux_cm2s,
                                          double next_growth_flux_cm2s);

// Derive scheduler-facing source policy for a recipe row.
// This function intentionally belongs in helpers rather than EffusionCell so
// that recipe semantics stay separate from source thermal state evolution.
//
// Arguments:
// - phase_code: explicit recipe phase for the current row
// - row_flux_cm2s: current row flux value
// - mbe_on: current row beam permission flag
// - next_growth_flux_cm2s: optional future growth anchor used by beam-off
//   pre-growth phases such as SOURCE_DEGAS and SOAK
SourcePhasePolicy deriveSourcePhasePolicy(PhaseCode phase_code,
                                          double row_flux_cm2s,
                                          int mbe_on,
                                          double next_growth_flux_cm2s = 0.0);

// Convenience predicates that let the scheduler reason about source behavior
// without repeating phase switch logic in main.cpp.
bool phaseUsesSourceControl(PhaseCode code);
bool phaseRequiresSourceReadiness(PhaseCode code);
bool phaseAllowsBeam(PhaseCode code);

// -----------------------------------------------------------------------------
// Job parsing and validation helpers
// -----------------------------------------------------------------------------
// Parse a single schedule line.
//
// Supported formats:
// - 8-column recipe-aware format
// - 4-column legacy format
//
// Returns true on success.
// Returns false on failure and fills err with a human-readable reason.
//
// Legacy rows are upgraded using defaults:
// - positive-flux rows become GROWTH rows with beam and substrate enabled
// - non-positive-flux rows become IDLE rows with beam and substrate disabled
bool parseJobLine(const std::string& line, Job& job, std::string& err);

// Validate semantic consistency of a parsed job row.
// Returns true if valid.
// Returns false and fills err if invalid.
bool validateJob(const Job& job, std::string& err);

// -----------------------------------------------------------------------------
// Duration helpers
// -----------------------------------------------------------------------------
// Total phase duration represented by the row. This is independent of whether
// beam-on growth occurs during the row.
int derivePhaseDurationTicks(const Job& job);

// Beam-on live deposition duration owed by the row. This is only nonzero for
// growth-like rows with positive flux and beam permission enabled.
int deriveLiveDepositionTicks(const Job& job);

// -----------------------------------------------------------------------------
// params.inc writer
// -----------------------------------------------------------------------------
// Rewrite inputDir/params.inc with the latest SPARTA-facing parameters.
//
// MPI behavior:
// - only rank 0 writes the file
// - this function performs no collectives and no barrier
// - synchronization remains the caller's responsibility
void write_params_inc(double Fwafer_cm2s,
                      double mbe_active,
                      int rank,
                      const std::string& inputDir,
                      LogFn log_fn);

} // namespace SimHelpers