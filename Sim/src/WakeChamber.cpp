#include "WakeChamber.hpp"
#include "SpartaBridge.hpp"   // <-- complete type visible here

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

WakeChamber::WakeChamber(MPI_Comm comm)
  : comm_(comm) {}

WakeChamber::~WakeChamber() = default;  // <-- defined where SpartaBridge is complete

void WakeChamber::init(const std::string& deck_basename,
                       const std::string& input_subdir) {
  if (initialized_) return;

  deck_         = deck_basename;
  input_subdir_ = input_subdir;

  // Open a dedicated SPARTA instance on this communicator
  sp_ = std::make_unique<SpartaBridge>(comm_);
  initialized_ = true;

  // First-time run to create grids, etc.
  sp_->runDeck(deck_, input_subdir_);
}

void WakeChamber::step() {
  if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");
  // Stateless smoke test: re-run the deck. Later: switch to sparta_command("run N").
  sp_->runDeck(deck_, input_subdir_);
}

void WakeChamber::shutdown() {
  sp_.reset();        // unique_ptr deletes after SpartaBridge is complete (OK here)
  initialized_ = false;
}

void WakeChamber::setParameter(const std::string& name, double value) {
  // Example: write a tiny include file that your deck reads each tick.
  // In your SPARTA deck, add at top:  include params.inc
  fs::path params = fs::path(PROJECT_SOURCE_DIR) / input_subdir_ / "params.inc";
  int rank = 0; MPI_Comm_rank(comm_, &rank);
  if (rank == 0) {
    std::ofstream out(params);
    out << "variable " << name << " equal " << value << "\n";
  }
  MPI_Barrier(comm_);
}
