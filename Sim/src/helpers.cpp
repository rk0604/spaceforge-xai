#include "helpers.hpp"

#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>

namespace SimHelpers {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const double FWAFFER_FLOOR_CM2S = 1.0e8;
const double DEFAULT_IDLE_SUBSTRATE_TARGET_K = 300.0;

// -----------------------------------------------------------------------------
// Tiny CLI helpers
// -----------------------------------------------------------------------------
static bool arg_eq(const char* a, const char* b) {
  return std::strcmp(a, b) == 0;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--mode") && i + 1 < argc)              a.mode = argv[++i];
    else if (arg_eq(argv[i], "--wake-deck") && i + 1 < argc)    a.wakeDeck = argv[++i];
    else if (arg_eq(argv[i], "--eff-deck") && i + 1 < argc)     a.effDeck  = argv[++i];
    else if (arg_eq(argv[i], "--input-subdir") && i + 1 < argc) a.inputDir = argv[++i];
    else if (arg_eq(argv[i], "--split") && i + 1 < argc)        a.nWake = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--couple-every") && i + 1 < argc) a.coupleEvery = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--sparta-block") && i + 1 < argc) a.spartaBlock = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--nticks") && i + 1 < argc)       a.nticks = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--dt") && i + 1 < argc)           a.dt = std::atof(argv[++i]);
    else if (arg_eq(argv[i], "--help"))                         a.showHelp = true;
  }
  return a;
}

void print_usage() {
  std::cout
    << "Usage: sim [--mode dual|legacy|wake|power]\n"
    << "           [--wake-deck in.wake_harness]\n"
    << "           [--input-subdir input]\n"
    << "           [--couple-every T] [--sparta-block N]\n"
    << "           [--nticks N] [--dt seconds]\n"
    << "\n"
    << "Modes:\n"
    << "  legacy  - single SPARTA instance on MPI_COMM_WORLD\n"
    << "  wake    - wake-only with in.wake_harness\n"
    << "  dual    - currently an alias of wake\n"
    << "  power   - C++ power and thermal harness only\n"
    << "\n"
    << "Coupling advances SPARTA by N steps every T engine ticks.\n";
}

// -----------------------------------------------------------------------------
// Phase helpers
// -----------------------------------------------------------------------------
const char* phaseCodeName(PhaseCode code) {
  switch (code) {
    case PhaseCode::IDLE:         return "IDLE";
    case PhaseCode::SOURCE_DEGAS: return "SOURCE_DEGAS";
    case PhaseCode::OXIDE_DESORB: return "OXIDE_DESORB";
    case PhaseCode::SOAK:         return "SOAK";
    case PhaseCode::NUCLEATE:     return "NUCLEATE";
    case PhaseCode::GROWTH:       return "GROWTH";
    case PhaseCode::ANNEAL:       return "ANNEAL";
    case PhaseCode::COOLDOWN:     return "COOLDOWN";
    default:                      return "UNKNOWN";
  }
}

PhaseCode parsePhaseCode(const std::string& text) {
  if (text == "IDLE")         return PhaseCode::IDLE;
  if (text == "SOURCE_DEGAS") return PhaseCode::SOURCE_DEGAS;
  if (text == "OXIDE_DESORB") return PhaseCode::OXIDE_DESORB;
  if (text == "SOAK")         return PhaseCode::SOAK;
  if (text == "NUCLEATE")     return PhaseCode::NUCLEATE;
  if (text == "GROWTH")       return PhaseCode::GROWTH;
  if (text == "ANNEAL")       return PhaseCode::ANNEAL;
  if (text == "COOLDOWN")     return PhaseCode::COOLDOWN;

  std::ostringstream oss;
  oss << "unknown phase_code: " << text;
  throw std::runtime_error(oss.str());
}

bool isGrowthPhase(PhaseCode code) {
  return (code == PhaseCode::NUCLEATE || code == PhaseCode::GROWTH);
}

bool isNonGrowthTimedPhase(PhaseCode code) {
  switch (code) {
    case PhaseCode::SOURCE_DEGAS:
    case PhaseCode::OXIDE_DESORB:
    case PhaseCode::SOAK:
    case PhaseCode::ANNEAL:
    case PhaseCode::COOLDOWN:
      return true;
    default:
      return false;
  }
}

// -----------------------------------------------------------------------------
// Flux-to-heater abstraction
// -----------------------------------------------------------------------------
double fluxToHeaterPower(double Fwafer_cm2s) {
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
    return 0.0;
  }

  const double F_low  = 5.0e13;
  const double F_high = 1.0e14;

  const double P_low  = 1200.0;
  const double P_high = 1800.0;

  double F = Fwafer_cm2s;
  if (F < F_low)  F = F_low;
  if (F > F_high) F = F_high;

  const double scale = (F - F_low) / (F_high - F_low);
  double P = P_low + scale * (P_high - P_low);

  if (P < 0.0)    P = 0.0;
  if (P > 2000.0) P = 2000.0;

  return P;
}

// -----------------------------------------------------------------------------
// Internal parsing helpers
// -----------------------------------------------------------------------------
static void normalizeTicks(Job& job) {
  if (job.end_tick < job.start_tick) {
    std::swap(job.start_tick, job.end_tick);
  }
}

static void applyLegacyDefaults(Job& job) {
  const bool positive_flux = std::isfinite(job.Fwafer_cm2s) && (job.Fwafer_cm2s > 0.0);

  if (positive_flux) {
    job.mbe_on = 1;
    job.substrate_on = 1;
    job.phase_code = PhaseCode::GROWTH;
    job.substrate_target_K = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
  } else {
    job.mbe_on = 0;
    job.substrate_on = 0;
    job.phase_code = PhaseCode::IDLE;
    job.substrate_target_K = DEFAULT_IDLE_SUBSTRATE_TARGET_K;
  }
}

bool parseJobLine(const std::string& line, Job& job, std::string& err) {
  err.clear();

  std::istringstream iss8(line);
  Job parsed8{};
  std::string phase_text;

  if (iss8 >> parsed8.start_tick
           >> parsed8.end_tick
           >> parsed8.Fwafer_cm2s
           >> parsed8.heater_W
           >> parsed8.mbe_on
           >> parsed8.substrate_on
           >> phase_text
           >> parsed8.substrate_target_K) {
    try {
      parsed8.phase_code = parsePhaseCode(phase_text);
    } catch (const std::exception& e) {
      err = e.what();
      return false;
    }

    normalizeTicks(parsed8);

    if (!validateJob(parsed8, err)) {
      return false;
    }

    job = parsed8;
    return true;
  }

  std::istringstream iss4(line);
  Job parsed4{};

  if (iss4 >> parsed4.start_tick
           >> parsed4.end_tick
           >> parsed4.Fwafer_cm2s
           >> parsed4.heater_W) {
    normalizeTicks(parsed4);
    applyLegacyDefaults(parsed4);

    if (!validateJob(parsed4, err)) {
      return false;
    }

    job = parsed4;
    return true;
  }

  err = "line does not match the 8-column or 4-column job format";
  return false;
}

bool validateJob(const Job& job, std::string& err) {
  err.clear();

  if (!std::isfinite(job.Fwafer_cm2s)) {
    err = "wafer_flux_cm2s must be finite";
    return false;
  }

  if (!std::isfinite(job.heater_W) || job.heater_W < 0.0) {
    err = "heater_W must be finite and non-negative";
    return false;
  }

  if (job.mbe_on != 0 && job.mbe_on != 1) {
    err = "mbe_on must be 0 or 1";
    return false;
  }

  if (job.substrate_on != 0 && job.substrate_on != 1) {
    err = "substrate_on must be 0 or 1";
    return false;
  }

  if (!std::isfinite(job.substrate_target_K) || job.substrate_target_K <= 0.0) {
    err = "substrate_target_K must be finite and positive";
    return false;
  }

  if (job.end_tick < job.start_tick) {
    err = "end_tick must be greater than or equal to start_tick";
    return false;
  }

  const bool positive_flux = (job.Fwafer_cm2s > 0.0);
  const bool growth_phase = isGrowthPhase(job.phase_code);
  const bool non_growth_timed_phase = isNonGrowthTimedPhase(job.phase_code);

  if (growth_phase && job.mbe_on == 0) {
    err = "growth-like phases require mbe_on = 1";
    return false;
  }

  if (growth_phase && !positive_flux) {
    err = "growth-like phases require positive wafer_flux_cm2s";
    return false;
  }

  if (non_growth_timed_phase && job.mbe_on != 0) {
    err = "non-growth timed phases require mbe_on = 0";
    return false;
  }

  if ((job.phase_code == PhaseCode::OXIDE_DESORB ||
       job.phase_code == PhaseCode::SOAK ||
       job.phase_code == PhaseCode::ANNEAL ||
       job.phase_code == PhaseCode::COOLDOWN) &&
      job.substrate_on == 0) {
    err = "thermal hold phases require substrate_on = 1";
    return false;
  }

  if (job.phase_code == PhaseCode::IDLE && job.mbe_on != 0) {
    err = "IDLE phase requires mbe_on = 0";
    return false;
  }

  if (job.phase_code == PhaseCode::IDLE && job.Fwafer_cm2s > 0.0) {
    err = "IDLE phase requires non-positive wafer_flux_cm2s";
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
// Duration helpers
// -----------------------------------------------------------------------------
int derivePhaseDurationTicks(const Job& job) {
  int raw = job.end_tick - job.start_tick;
  if (raw < 0) raw = 0;

  // Every explicit row represents a real timed phase.
  // Zero-flux rows are not placeholders.
  return std::max(1, raw);
}

int deriveLiveDepositionTicks(const Job& job) {
  int raw = job.end_tick - job.start_tick;
  if (raw < 0) raw = 0;

  const bool positive_flux = std::isfinite(job.Fwafer_cm2s) && (job.Fwafer_cm2s > 0.0);
  const bool beam_enabled  = (job.mbe_on != 0);
  const bool growth_like   = isGrowthPhase(job.phase_code);

  if (positive_flux && beam_enabled && growth_like) {
    return std::max(1, raw);
  }

  return 0;
}

// -----------------------------------------------------------------------------
// params.inc writer
// -----------------------------------------------------------------------------
void write_params_inc(double Fwafer_cm2s,
                      double mbe_active,
                      int rank,
                      const std::string& inputDir,
                      LogFn log_fn) {
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
    Fwafer_cm2s = FWAFFER_FLOOR_CM2S;
  }

  if (!std::isfinite(mbe_active)) {
    mbe_active = 0.0;
  }
  mbe_active = (mbe_active > 0.5) ? 1.0 : 0.0;

  if (rank != 0) {
    return;
  }

  const std::string path = inputDir + "/params.inc";

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::ostringstream oss;
    oss << "[fatal] Cannot open " << path << " for writing.\n";
    if (log_fn) log_fn(oss.str());
    throw std::runtime_error("failed to open params.inc for writing");
  }

  out << "variable Fwafer_cm2s  equal " << Fwafer_cm2s << "\n";
  out << "variable mbe_active   equal " << mbe_active  << "\n";
  out.flush();

  if (!out) {
    std::ostringstream oss;
    oss << "[fatal] Failed while writing " << path << ".\n";
    if (log_fn) log_fn(oss.str());
    throw std::runtime_error("failed while writing params.inc");
  }

  out.close();

  std::ostringstream oss;
  oss << "[params] Wrote params.inc: Fwafer_cm2s=" << Fwafer_cm2s
      << ", mbe_active=" << mbe_active << "\n";
  if (log_fn) log_fn(oss.str());
}

} // namespace SimHelpers