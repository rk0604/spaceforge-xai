// Sim/src/main.cpp
// mpirun -np 4 ./build/Sim/sim_app
/**
Build (from repo root):
  cd ~/spaceforge-xai
  rm -rf build && mkdir build && cd build
  cmake -DSPARTA_DIR="$HOME/opt/sparta/src" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . -j

Run (from build/, headless):
  env -u DISPLAY mpirun -np 4 ./Sim/sim_app                            # dual-instance (default)
  env -u DISPLAY mpirun -np 4 ./Sim/sim_app --mode legacy              # original single-instance
  env -u DISPLAY mpirun -np 4 ./Sim/sim_app --mode dual --split 3      # 3 ranks wake, 1 effusion
  env -u DISPLAY mpirun -np 4 ./Sim/sim_app --mode dual \
    --wake-deck in.wake --eff-deck in.effusion --input-subdir input \
    --couple-every 10 --sparta-block 200
*/

#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"
#include "WakeChamber.hpp"

// ---------------------------
// Tiny CLI helpers (no deps)
// ---------------------------
struct Args {
  std::string mode     = "dual";     // "dual" or "legacy"
  std::string wakeDeck = "in.wake";
  std::string effDeck  = "in.effusion";
  std::string inputDir = "input";
  int  nWake           = -1;         // -1 => world_size/2
  int  coupleEvery     = 10;         // advance SPARTA every X engine ticks
  int  spartaBlock     = 200;        // run N steps per advance
  bool showHelp        = false;
};

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a,b) == 0; }

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (arg_eq(argv[i], "--mode") && i+1 < argc)             a.mode = argv[++i];
    else if (arg_eq(argv[i], "--wake-deck") && i+1 < argc)   a.wakeDeck = argv[++i];
    else if (arg_eq(argv[i], "--eff-deck") && i+1 < argc)    a.effDeck  = argv[++i];
    else if (arg_eq(argv[i], "--input-subdir") && i+1 < argc)a.inputDir = argv[++i];
    else if (arg_eq(argv[i], "--split") && i+1 < argc)       a.nWake = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--couple-every") && i+1 < argc)a.coupleEvery = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--sparta-block") && i+1 < argc)a.spartaBlock = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--help"))                      a.showHelp = true;
  }
  return a;
}

static void print_usage() {
  std::cout <<
    "Usage: sim_app [--mode dual|legacy]\n"
    "               [--wake-deck in.wake] [--eff-deck in.effusion]\n"
    "               [--input-subdir input] [--split N]\n"
    "               [--couple-every T] [--sparta-block N]\n"
    "Default 'dual' splits MPI ranks into wake+effusion; each runs persistently.\n"
    "Coupling advances SPARTA by N steps every T engine ticks.\n";
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Args args = parse_args(argc, argv);
  if (args.showHelp) {
    if (rank == 0) print_usage();
    MPI_Finalize();
    return 0;
  }

  try {
    // ------------------------------------------------------------------------
    // Electrical/power subsystems (independent of SPARTA)
    // ------------------------------------------------------------------------
    PowerBus   bus;
    SolarArray solar;
    Battery    battery;
    HeaterBank heater(/*maxPowerW=*/200.0);

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);

    SimulationEngine engine;
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    const double dt = 0.1; // seconds per tick
    engine.setTickStep(dt);
    engine.initialize();

    if (rank == 0) {
      std::cout << "Simulation starting on " << size
                << " MPI task(s), dt=" << dt << " s\n";
      std::cout << "Mode = " << args.mode << "\n";
    }

    // ======================================================================
    // MODE A: legacy (single SPARTA instance on WORLD)
    // ======================================================================
    if (args.mode == "legacy") {
      WakeChamber wake(MPI_COMM_WORLD);
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());

      const int NTICKS = 500;
      for (int i = 0; i < NTICKS; ++i) {
        heater.setDemand(150.0);
        engine.tick();

        if (i % args.coupleEvery == 0) {
          // Persistent advance without re-reading input
          wake.runIfDirtyOrAdvance(args.spartaBlock);
        }
      }
      wake.shutdown();
      engine.shutdown();
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      return EXIT_SUCCESS;
    }

    // ======================================================================
    // MODE B: dual (recommended) -- split communicators
    // ======================================================================
    int nWake = (args.nWake > 0 ? args.nWake : size / 2);
    if (nWake < 1) nWake = 1;
    if (nWake > size - 1) nWake = size - 1;
    if (size == 1) nWake = 1;

    int color = (rank < nWake) ? 0 : 1; // 0=Wake, 1=Effusion
    MPI_Comm subcomm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);

    if (rank == 0) {
      std::cout << "Dual-instance split: nWake=" << nWake
                << ", nEffusion=" << (size - nWake) << "\n";
    }

    if (color == 0) {
      // ------------------------ Wake Chamber ranks --------------------------
      WakeChamber wake(subcomm);
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());

      const int NTICKS = 500;
      for (int i = 0; i < NTICKS; ++i) {
        heater.setDemand(150.0);
        engine.tick();

        // Advance SPARTA only at coupling cadence; persistent instance
        if (i % args.coupleEvery == 0) {
          wake.runIfDirtyOrAdvance(args.spartaBlock);
        }
      }
      wake.shutdown();

    } else {
      // ------------------------ Effusion Cell ranks -------------------------
      WakeChamber eff(subcomm);
      eff.init(args.effDeck.c_str(), args.inputDir.c_str());

      const int NTICKS = 500;
      for (int i = 0; i < NTICKS; ++i) {
        // If parameters change significantly, call eff.markDirtyReload();
        if (i % args.coupleEvery == 0) {
          eff.runIfDirtyOrAdvance(args.spartaBlock);
        }
      }
      eff.shutdown();
    }

    MPI_Comm_free(&subcomm);
    engine.shutdown();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }
  catch (const std::exception& e) {
    std::cerr << "[fatal] " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return EXIT_FAILURE;
  }
}
