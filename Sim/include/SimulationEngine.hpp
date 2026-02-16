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

    void markJobFailedThisTick();

    // Set spacecraft housekeeping power draw (W)
    void setBaseLoadW(double w);

private:
    std::vector<Subsystem*> subsystems_;
    TickPhaseEngine tickEngine_;

    Battery*    battery_   = nullptr;
    SolarArray* solar_     = nullptr;
    PowerBus*   powerbus_  = nullptr;
    HeaterBank* heater_    = nullptr;

    int    tick_count_ = 0;
    double sim_time_   = 0.0;
    double tick_step_  = 60.0;

    bool job_failed_flag_ = false;

    // Spacecraft baseline power draw
    double base_load_W_  = 400.0; // commanded housekeeping draw
    double base_drawn_W_ = 0.0;   // actually granted by bus/battery this tick

    // ---- Latched values captured BEFORE PowerBus::tick() resets counters ----
    double latched_bus_remaining_W_   = 0.0;
    double latched_total_requested_W_ = 0.0;
    double latched_total_granted_W_   = 0.0;
    double latched_total_generated_W_ = 0.0;

    void logRow_(int tick, double time);
};
