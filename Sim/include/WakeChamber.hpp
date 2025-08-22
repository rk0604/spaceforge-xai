#pragma once
#include <string>
#include <memory>
#include <mpi.h>

class SpartaBridge;  // forward declaration

// A thin façade around a SPARTA instance dedicated to wake modeling.
class WakeChamber {
public:
  explicit WakeChamber(MPI_Comm comm);
  ~WakeChamber();  // <-- declare only; define in .cpp

  // Open SPARTA and prepare the deck. Defaults point to:
  //   <project>/input/in.wake  and relative includes under <project>/input/data/
  void init(const std::string& deck_basename = "in.wake",
            const std::string& input_subdir  = "input");

  // Advance the wake simulation by one "tick".
  void step();

  // Close the SPARTA instance.
  void shutdown();

  // Optional: inject parameters before next step by writing an include file.
  void setParameter(const std::string& name, double value);

private:
  MPI_Comm                            comm_;
  std::unique_ptr<SpartaBridge>       sp_;
  std::string                         deck_;
  std::string                         input_subdir_;
  bool                                initialized_ = false;
};
