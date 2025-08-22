// Sim/src/main.cpp
#include <mpi.h>
#include <cstdlib>
#include <iostream>

#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"

// ---------------------------
// SMOKE TEST ADDITIONS
// ---------------------------
// Bring in the WakeChamber façade which internally talks to SPARTA
#include "WakeChamber.hpp"

int main(int argc, char** argv) {              // FIX 1: take argc/argv
  // FIX 2: initialize MPI with the actual argc/argv pointers
  // (You can switch to MPI_Init_thread if you later use threads.)
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    // --- Subsystems ----------------------------------------------------------
    PowerBus   bus;
    SolarArray solar;
    Battery    battery;
    HeaterBank heater(/*maxPowerW=*/200.0);     // can draw up to 200 W

    // Wire connections
    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);

    // --- Engine setup --------------------------------------------------------
    SimulationEngine engine;
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    const double dt = 0.1;                      // seconds per tick
    engine.setTickStep(dt);
    engine.initialize();

    if (rank == 0) {
      std::cout << "Simulation starting on " << size
                << " MPI task(s), dt=" << dt << " s\n";
    }

    // ========================================================================
    // SMOKE TEST: WakeChamber + SPARTA
    // ------------------------------------------------------------------------
    // We keep a live WakeChamber instance (backed by SPARTA) while the
    // electrical/power subsystems tick. For now we run it on WORLD; later you
    // can split ranks with MPI_Comm_split to dedicate a sub-communicator.
    // - init() will load and run the deck once to set up grids/state.
    // - step() currently re-runs the deck each tick (simple, stateless demo).
    //   Once you add a persistent deck, swap this for a "run N" command.
    // ========================================================================
    WakeChamber wake(MPI_COMM_WORLD);
    wake.init(/*deck_basename*/ "in.wake",
              /*input_subdir*/  "input");

    // --- Main loop -----------------------------------------------------------
    const int NTICKS = 50;
    for (int i = 0; i < NTICKS; ++i) {
      heater.setDemand(150.0);                  // request 150 W each tick
      engine.tick();                            // engine handles parallel dispatch

      // ---- SMOKE TEST step: advance the wake solver this tick --------------
      wake.step();
    }

    // Tidy up the SPARTA-backed module first (optional order)
    wake.shutdown();

    engine.shutdown();

    // Be tidy: synchronize before finalize
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }
  catch (const std::exception& e) {
    // FIX 3: if anything throws, abort the whole MPI job to avoid hangs
    std::cerr << "[fatal] " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return EXIT_FAILURE;
  }
}
