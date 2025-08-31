#include "WakeChamber.hpp"
#include "SpartaBridge.hpp"    // complete type visible here

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

WakeChamber::WakeChamber(MPI_Comm comm)
  : comm_(comm) {}

WakeChamber::~WakeChamber() = default;

void WakeChamber::init(const std::string& deck_basename,
                       const std::string& input_subdir) {
  if (initialized_) return;

  deck_         = deck_basename;
  input_subdir_ = input_subdir;

  // Create a dedicated SPARTA instance on this communicator
  sp_ = std::make_unique<SpartaBridge>(comm_);

  // Read the input script ONCE (creates box/grid/etc.)
  sp_->runDeck(deck_, input_subdir_);

  initialized_  = true;
  dirtyReload_  = false;
}

void WakeChamber::step(int nDefault) {
  if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");
  // Do NOT re-read the deck. Just advance by a block of steps.
  if (nDefault > 0) runSteps(nDefault);
}

void WakeChamber::runSteps(int n) {
  if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");
  if (n <= 0) return;
  // Issue a plain "run N" to SPARTA
  std::string cmd = "run " + std::to_string(n);
  sp_->command(cmd.c_str());
}

void WakeChamber::markDirtyReload() {
  dirtyReload_ = true;
}

bool WakeChamber::runIfDirtyOrAdvance(int n) {
  if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");

  if (dirtyReload_) {
    // Full reset + re-read the original deck (e.g., geometry/topology changed)
    sp_->command("clear");
    sp_->runDeck(deck_, input_subdir_);
    dirtyReload_ = false;
    // Optional: still advance after reload
    if (n > 0) runSteps(n);
    return true;
  } else {
    // No reload needed—just advance
    if (n > 0) runSteps(n);
    return (n > 0);
  }
}

void WakeChamber::shutdown() {
  sp_.reset();               // destroy SPARTA instance
  initialized_ = false;
  dirtyReload_ = false;
  deck_.clear();
  input_subdir_.clear();
}

void WakeChamber::setParameter(const std::string& name, double value) {
  // Example: write a tiny include file that your deck reads.
  // In the SPARTA deck, add near the top:   include params.inc
  fs::path params = fs::path(PROJECT_SOURCE_DIR) / input_subdir_ / "params.inc";

  int rank = 0;
  MPI_Comm_rank(comm_, &rank);
  if (rank == 0) {
    std::ofstream out(params);
    if (!out) throw std::runtime_error("WakeChamber::setParameter: cannot open params.inc");
    out << "variable " << name << " equal " << value << "\n";
  }
  MPI_Barrier(comm_);
}
