/*
Minimal SPARTA runner (external-library link)
---------------------------------------------
Build SPARTA once (external):
  mkdir -p ~/opt && cd ~/opt
  git clone https://github.com/sparta/sparta.git
  cd sparta/src
  make -j8 mpi mode=lib CXX=$(which mpicxx) CC=$(which mpicc)
  # => ~/opt/sparta/src/libsparta_mpi.a  (and symlink libsparta.a)

Build this project:
  cd ~/spaceforge-xais
  cmake -S . -B build -DMPI_CXX_COMPILER=$(which mpicxx) -DSPARTA_DIR=$HOME/opt/sparta/src
  cmake --build build -j

Run:
  # from the deck directory so relative paths work
  cd ~/opt/sparta/examples/collide
  DISPLAY= mpirun -np 4 ~/spaceforge_sim/build/simulation in.collide

  Note: in.collide is a deck (a plain-text SPARTA input script )
         - list them all using: find ~/opt/sparta -type f -name 'in.*' | sort
         - peak inside a deck using: head -n 40 ~/opt/sparta/examples/collide/in.collide
*/

#include <iostream>
#include <mpi.h>
#include "library.h"  // SPARTA C API

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Properly define argv for SPARTA (mutable char arrays)
  static char prog[] = "sparta";
  static char opt1[] = "-log";
  static char opt2[] = "log.capi";
  char* sp_argv[] = { prog, opt1, opt2 };
  int   sp_argc   = sizeof(sp_argv) / sizeof(sp_argv[0]);

  void* spa = nullptr;
  sparta_open(sp_argc, sp_argv, MPI_COMM_WORLD, &spa);
  if (!spa) {
    if (rank == 0) std::cerr << "Failed to open SPARTA library.\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // const char* deck = (argc > 1) ? argv[1] : "in.demo";
  // if (rank == 0) std::cout << "Running SPARTA deck: " << deck << std::endl;

    // Default to in.wake and cd into input/ so relative paths (data/ar.*) work
  fs::path input_dir = fs::path(PROJECT_SOURCE_DIR) / "input";
  fs::current_path(input_dir);

  const char* deck = (argc > 1) ? argv[1] : "in.wake";
  if (rank==0) std::cout << "Running SPARTA deck: " << deck
                         << " from " << fs::current_path().string() << "\n";

  // Run the deck
  sparta_file(spa, const_cast<char*>(deck));

  sparta_close(spa);
  MPI_Finalize();
  return 0;
}
