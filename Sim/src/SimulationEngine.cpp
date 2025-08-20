#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "PowerBus.hpp"
#include "SolarArray.hpp"
#include "HeaterBank.hpp"
#include "Logger.hpp"
#include "TickPhaseEngine.hpp"

void SimulationEngine::addSubsystem(Subsystem* subsystem) {
    subsystems_.push_back(subsystem);
}

void SimulationEngine::initialize() {
    // link core subsystems if present
    for (auto* s : subsystems_) {
        if (auto* b = dynamic_cast<Battery*>(s)) battery_ = b;
        else if (auto* sa = dynamic_cast<SolarArray*>(s)) solar_ = sa;
        else if (auto* pb = dynamic_cast<PowerBus*>(s)) powerbus_ = pb;
        else if (auto* hb = dynamic_cast<HeaterBank*>(s)) heater_ = hb;

        s->initialize();
        tickEngine_.addSubsystem(s);   // register with TickPhaseEngine
    }

    tickEngine_.start();  // launch worker threads
}

void SimulationEngine::tick() {
    TickContext ctx {
        .tick_index = tick_count_,
        .time = sim_time_,
        .dt = tick_step_
    };

    // run all subsystems in parallel
    tickEngine_.runTick(ctx);

    Logger::instance().log("SimulationEngine", tick_count_, sim_time_, {
        {"battery", battery_ ? battery_->getCharge() : -1},
        {"solar",   solar_   ? solar_->getLastOutput() : -1},
        {"bus",     powerbus_? powerbus_->getAvailablePower() : -1}
    });

    tick_count_++;
    sim_time_ += tick_step_;
}

void SimulationEngine::setTickStep(double dt) {
    tick_step_ = dt;
}

void SimulationEngine::shutdown() {
    tickEngine_.stop();
    for (auto* s : subsystems_) s->shutdown();
}
