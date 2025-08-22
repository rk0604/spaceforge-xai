#pragma once
#include <string>
#include <mpi.h>

class SpartaBridge {
public:
  explicit SpartaBridge(MPI_Comm comm = MPI_COMM_WORLD);
  ~SpartaBridge();

  // Runs a deck from PROJECT_SOURCE_DIR/<input_subdir>
  void runDeck(const std::string& deck_basename,
               const std::string& input_subdir = "input");

private:
  void* spa_ = nullptr;
  MPI_Comm comm_;
};
