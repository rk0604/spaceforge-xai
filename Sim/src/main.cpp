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
#include "EffusionCell.hpp"   

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

  // make tick size and run length configurable
  int    nticks        = 500;        // engine ticks to run
  double dt            = 0.1;        // seconds per engine tick
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
    else if (arg_eq(argv[i], "--nticks") && i+1 < argc)      a.nticks = std::atoi(argv[++i]);
    else if (arg_eq(argv[i], "--dt") && i+1 < argc)          a.dt = std::atof(argv[++i]);
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
    "               [--nticks N] [--dt seconds]\n"
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
    // NOTE: We host the Power/Heater/EffusionCell engine on the wake side.
    // It still gets constructed here for simplicity; only the wake leader ticks it.
    // ------------------------------------------------------------------------
    PowerBus   bus;
    SolarArray solar;
    Battery    battery;
    HeaterBank heater(/*maxPowerW=*/200.0);

    EffusionCell effCell;                 

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);
    heater.setEffusionCell(&effCell);     // Heater warms the effusion cell

    SimulationEngine engine;
    engine.addSubsystem(&effCell); 
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    const double dt = args.dt;
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
      // give this instance a stable label so it logs to WakeChamber.csv
      WakeChamber wake(MPI_COMM_WORLD, "WakeChamber");
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());

      int worldrank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        if (worldrank == 0) {             // only leader ticks/logs per-tick subsystems
          heater.setDemand(150.0);
          engine.tick();
        }

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

    // define a tiny coupling channel between leaders
    const int TAG_CELL_TEMP = 1001;
    const int world_leader_wake = 0;        // first wake rank in WORLD
    const int world_leader_eff  = nWake;    // first effusion rank in WORLD

    if (color == 0) {
      // ------------------------ Wake Chamber ranks --------------------------
      // label this SPARTA instance so it logs to data/raw/WakeChamber.csv
      WakeChamber wake(subcomm, "WakeChamber");
      wake.init(args.wakeDeck.c_str(), args.inputDir.c_str());

      int subrank = 0;
      MPI_Comm_rank(subcomm, &subrank);

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        if (subrank == 0) {               // only wake leader ticks/logs
          heater.setDemand(150.0);
          engine.tick();
        }

        // Advance SPARTA only at coupling cadence; persistent instance
        if (i % args.coupleEvery == 0) {
          wake.runIfDirtyOrAdvance(args.spartaBlock);

          // send effusion cell temperature to effusion leader
          if (subrank == 0) {
            double T = effCell.getTemperature();
            MPI_Send(&T, 1, MPI_DOUBLE, world_leader_eff, TAG_CELL_TEMP, MPI_COMM_WORLD);
          }
        }
      }
      wake.shutdown();

    } else {
      // ------------------------ Effusion Cell ranks -------------------------
      // distinct label so it logs to data/raw/EffusionChamber.csv
      WakeChamber eff(subcomm, "EffusionChamber");
      eff.init(args.effDeck.c_str(), args.inputDir.c_str());

      int subrank = 0;
      MPI_Comm_rank(subcomm, &subrank);

      const int NTICKS = args.nticks;
      for (int i = 0; i < NTICKS; ++i) {
        // At the coupling cadence, receive latest temperature from wake leader,
        // push it into SPARTA via params.inc, then advance.
        if (i % args.coupleEvery == 0) {
          if (subrank == 0) {
            double T_recv = 0.0;
            MPI_Status st{};
            MPI_Recv(&T_recv, 1, MPI_DOUBLE, world_leader_wake, TAG_CELL_TEMP,
                     MPI_COMM_WORLD, &st);

            // map incoming temperature into effusion deck
            eff.setParameter("cell_temp_K", T_recv);
            eff.markDirtyReload();
          }
          // keep ranks in lockstep inside the effusion subcomm (params.inc barrier)
          MPI_Barrier(subcomm);

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
