#pragma once
#include <vector>
#include "Subsystem.hpp"
#include "TickPhaseEngine.hpp"
#include "Logger.hpp"

class Battery;
class SolarArray;
class PowerBus;
class HeaterBank;

class SimulationEngine {
public:
    void addSubsystem(Subsystem* subsystem);
    void initialize();
    void tick();
    void setTickStep(double dt);
    void shutdown();

private:
    std::vector<Subsystem*> subsystems_;
    TickPhaseEngine tickEngine_;   // kept for future use (not required here)

    Battery*    battery_   = nullptr;
    SolarArray* solar_     = nullptr;
    PowerBus*   powerbus_  = nullptr;
    HeaterBank* heater_    = nullptr;

    int    tick_count_ = 0;
    double sim_time_   = 0.0;
    double tick_step_  = 60.0;   // seconds per tick (match your CSV cadence)

    // helper to emit one wide row for the engine snapshot
    void logRow_(int tick, double time);
};
