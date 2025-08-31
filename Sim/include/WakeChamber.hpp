#pragma once
#include <string>
#include <memory>
#include <mpi.h>

class SpartaBridge;  // forward declaration

// Thin façade around a single SPARTA instance.
// - init() reads the deck once and keeps SPARTA alive
// - runSteps(N) advances without re-reading the deck (issues "run N")
// - markDirtyReload() -> next runIfDirtyOrAdvance() will clear+reload the deck
class WakeChamber {
public:
  explicit WakeChamber(MPI_Comm comm);
  ~WakeChamber();                                   // defined in .cpp

  // Open SPARTA and read the deck once.
  // Defaults resolve to:  <project>/input/in.wake
  void init(const std::string& deck_basename = "in.wake",
            const std::string& input_subdir  = "input");

  // Legacy convenience: advance by a default block of steps (no re-read).
  // Uses runSteps() under the hood.
  void step(int nDefault = 1000);

  // Close the SPARTA instance.
  void shutdown();

  // Persistent advance without re-reading the deck.
  void runSteps(int n);

  // Mark that a big change happened -> we must reload the deck next time.
  void markDirtyReload();

  // If dirty: clear+file(deck), else: run N. Returns true if it did something.
  bool runIfDirtyOrAdvance(int n);

  // Optional: inject parameters by writing an include file the deck reads.
  void setParameter(const std::string& name, double value);

private:
  MPI_Comm                      comm_;
  std::unique_ptr<SpartaBridge> sp_;
  std::string                   deck_;
  std::string                   input_subdir_;
  bool                          initialized_ = false;
  bool                          dirtyReload_ = false;
};
