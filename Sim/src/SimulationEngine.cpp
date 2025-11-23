#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "PowerBus.hpp"
#include "SolarArray.hpp"
#include "HeaterBank.hpp"
#include "Logger.hpp"

void SimulationEngine::addSubsystem(Subsystem* subsystem) {
    subsystems_.push_back(subsystem);
}

void SimulationEngine::initialize() {
    // Identify “well-known” subsystems and initialize all
    for (auto* s : subsystems_) {
        if (auto* b  = dynamic_cast<Battery*>(s))    battery_  = b;
        if (auto* sa = dynamic_cast<SolarArray*>(s)) solar_   = sa;
        if (auto* pb = dynamic_cast<PowerBus*>(s))   powerbus_ = pb;
        if (auto* hb = dynamic_cast<HeaterBank*>(s)) heater_   = hb;
    }

    for (auto* s : subsystems_) s->initialize();

    // No job has failed before the sim starts
    job_failed_flag_ = false;

    // Initial snapshot row
    logRow_(/*tick*/0, /*time*/0.0);

    tick_count_ = 1;   // next tick index
    sim_time_   = tick_step_;
}

void SimulationEngine::tick() {
    const TickContext ctx{ tick_count_, sim_time_, tick_step_ };

    // Advance all subsystems once
    for (auto* s : subsystems_) s->tick(ctx);

    // Snapshot the system state in one wide row.
    // job_failed_flag_ reflects any failure that was marked
    // since the previous tick. After logging, clear it so it
    // only shows up for a single row.
    logRow_(tick_count_, sim_time_);
    job_failed_flag_ = false;

    // Advance time
    tick_count_ += 1;
    sim_time_   += tick_step_;
}

void SimulationEngine::setTickStep(double dt) {
    tick_step_ = dt;
}

void SimulationEngine::markJobFailedThisTick() {
    // Called from main.cpp when a job is deemed failed.
    // This flag will appear as job_failed=1.0 in the *next*
    // SimulationEngine.csv row that gets logged.
    job_failed_flag_ = true;
}

void SimulationEngine::shutdown() {
    // No -1 sentinel row to keep CSVs consistent
    for (auto* s : subsystems_) s->shutdown();
}

void SimulationEngine::logRow_(int tick, double time) {
    double bus   = 0.0;
    double batt  = 0.0;
    double solar = 0.0;

    if (powerbus_) bus   = powerbus_->getAvailablePower();
    if (battery_)  batt  = battery_->getCharge();
    if (solar_)    solar = solar_->getLastOutput();

    const double job_failed = job_failed_flag_ ? 1.0 : 0.0;

    Logger::instance().log_wide(
        "SimulationEngine",
        tick,
        time,
        {"status","bus","battery","solar","job_failed"},
        {1.0,      bus,  batt,     solar,  job_failed}
    );
}
