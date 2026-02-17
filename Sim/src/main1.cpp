// Sim/src/main.cpp
#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include "App.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rc = EXIT_FAILURE;
  try {
    App app(argc, argv);
    rc = app.run();
  }
  catch (const std::exception& e) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "[fatal] std::exception: " << e.what() << "\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  catch (...) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "[fatal] Unknown non-std exception\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return rc;
}
