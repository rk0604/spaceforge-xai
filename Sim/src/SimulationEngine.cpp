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
        if (auto* b = dynamic_cast<Battery*>(s))    battery_  = b;
        if (auto* sa = dynamic_cast<SolarArray*>(s)) solar_   = sa;
        if (auto* pb = dynamic_cast<PowerBus*>(s))   powerbus_ = pb;
        if (auto* hb = dynamic_cast<HeaterBank*>(s)) heater_   = hb;
    }

    for (auto* s : subsystems_) s->initialize();

    // Initial snapshot row
    logRow_(/*tick*/0, /*time*/0.0);

    tick_count_ = 1;   // next tick index
    sim_time_   = tick_step_;
}

void SimulationEngine::tick() {
    const TickContext ctx{ tick_count_, sim_time_, tick_step_ };

    // Advance all subsystems once
    for (auto* s : subsystems_) s->tick(ctx);

    // Snapshot the system state in one wide row
    logRow_(tick_count_, sim_time_);

    // Advance time
    tick_count_ += 1;
    sim_time_   += tick_step_;
}

void SimulationEngine::setTickStep(double dt) {
    tick_step_ = dt;
}

void SimulationEngine::shutdown() {
    // No -1 sentinel row to keep CSVs consistent
    for (auto* s : subsystems_) s->shutdown();
}

void SimulationEngine::logRow_(int tick, double time) {
    double bus     = 0.0;
    double batt    = 0.0;
    double solar   = 0.0;

    if (powerbus_) bus  = powerbus_->getAvailablePower();
    if (battery_)  batt = battery_->getCharge();
    if (solar_)    solar = solar_->getLastOutput();

    Logger::instance().log_wide(
        "SimulationEngine",
        tick,
        time,
        {"status","bus","battery","solar"},
        {1.0, bus, batt, solar}
    );
}
