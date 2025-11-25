#pragma once

#include <string>
#include <vector>
#include <functional>

namespace SimHelpers {

// ---------------------------
// CLI arguments / config
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

  // tick size and run length
  int    nticks        = 500;        // engine ticks to run
  double dt            = 60;         // seconds per engine tick
};

// ---------------------------
// Job schedule for effusion
// ---------------------------
struct Job {
  int    start_tick = 0;      // inclusive
  int    end_tick   = -1;     // inclusive
  double Fwafer_cm2s = 0.0;   // effusion flux to send to SPARTA
  double heater_W    = 0.0;   // (legacy) heater demand in watts
};

// Global floor so Fwafer_cm2s never goes fully to zero in SPARTA mixture.
extern const double FWAFFER_FLOOR_CM2S;

// Logger function type used by helpers (implemented in main.cpp).
using LogFn = std::function<void(const std::string&)>;

// Argument helpers
Args parse_args(int argc, char** argv);
void print_usage();

// Physics helper
double fluxToHeaterPower(double Fwafer_cm2s);

// params.inc writer (wraps old lambda, keeps behavior identical)
void write_params_inc(double Fwafer_cm2s,
                      double mbe_active,
                      int rank,
                      const std::string& inputDir,
                      LogFn log_fn);

} // namespace SimHelpers
