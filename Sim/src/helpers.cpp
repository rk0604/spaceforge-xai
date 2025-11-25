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

namespace SimHelpers {

// Same floor value as before
const double FWAFFER_FLOOR_CM2S = 1.0e8;

// ---------------------------
// Tiny CLI helpers (no deps)
// ---------------------------
static bool arg_eq(const char* a, const char* b) {
  return std::strcmp(a, b) == 0;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--mode") && i + 1 < argc)              a.mode = argv[++i];
    else if (arg_eq(argv[i], "--wake-deck") && i + 1 < argc)    a.wakeDeck = argv[++i];
    else if (arg_eq(argv[i], "--eff-deck") && i + 1 < argc)     a.effDeck  = argv[++i];  // ignored now
    else if (arg_eq(argv[i], "--input-subdir") && i + 1 < argc) a.inputDir = argv[++i];
    else if (arg_eq(argv[i], "--split") && i + 1 < argc)        a.nWake = std::atoi(argv[++i]); // ignored
    else if (arg_eq(argv[i], "--couple-every") && i + 1 < argc) a.coupleEvery = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--sparta-block") && i + 1 < argc) a.spartaBlock = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--nticks") && i + 1 < argc)       a.nticks = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--dt") && i + 1 < argc)           a.dt = std::atof(argv[++i]);
    else if (arg_eq(argv[i], "--help"))                         a.showHelp = true;
  }
  return a;
}

void print_usage() {
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
// Map wafer flux to heater power
// ---------------------------
double fluxToHeaterPower(double Fwafer_cm2s) {
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

// ---------------------------
// params.inc writer
// ---------------------------
void write_params_inc(double Fwafer_cm2s,
                      double mbe_active,
                      int rank,
                      const std::string& inputDir,
                      LogFn log_fn) {
  // Clamp flux to positive floor
  if (!std::isfinite(Fwafer_cm2s) || Fwafer_cm2s <= 0.0) {
    Fwafer_cm2s = FWAFFER_FLOOR_CM2S;
  }
  if (!std::isfinite(mbe_active)) {
    mbe_active = 0.0;
  }

  if (rank == 0) {
    std::string path = inputDir + "/params.inc";
    std::ofstream out(path);
    if (!out) {
      std::ostringstream oss;
      oss << "[fatal] Cannot open " << path << " for writing.\n";
      if (log_fn) log_fn(oss.str());
      throw std::runtime_error("Failed to write params.inc");
    }
    out << "variable Fwafer_cm2s  equal " << Fwafer_cm2s  << "\n";
    out << "variable mbe_active   equal " << mbe_active   << "\n";
    out.close();

    std::ostringstream oss;
    oss << "[params] Wrote params.inc: Fwafer_cm2s=" << Fwafer_cm2s
        << ", mbe_active=" << mbe_active << "\n";
    if (log_fn) log_fn(oss.str());
  }

  // Make sure all ranks wait until file is written
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace SimHelpers
