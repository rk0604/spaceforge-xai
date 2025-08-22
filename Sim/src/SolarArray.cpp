#include "SolarArray.hpp"
#include "Logger.hpp"
#include <cmath>
#include <iostream>

SolarArray::SolarArray(double efficiency, double base_input)
    : Subsystem("SolarArray"),
      bus_(nullptr),
      efficiency_(efficiency),
      base_input_(base_input),
      last_output_(0.0) {}

void SolarArray::initialize() {
    last_output_ = 0.0;
    Logger::instance().log("SolarArray", 0, 0.0, {
        {"status", 1},
        {"efficiency", efficiency_}
    });
}

void SolarArray::tick(const TickContext& ctx) {
    // Simple model: sinusoidal solar input
    double solar_input = 1000.0 * std::fabs(std::sin(ctx.time));
    last_output_ = solar_input * efficiency_;

    if (bus_) {
        bus_->addPower(last_output_);
    }

    Logger::instance().log("SolarArray", ctx.tick_index, ctx.time, {
        {"solar_input", solar_input},
        {"output", last_output_}
    });
}

void SolarArray::shutdown() {
    Logger::instance().log("SolarArray", -1, -1.0, {
        {"status", 0}
    });
}

void SolarArray::setPowerBus(PowerBus* bus) {
    bus_ = bus;
}

double SolarArray::getLastOutput() const {
    return last_output_;
}
