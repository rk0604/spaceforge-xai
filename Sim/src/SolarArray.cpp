#include "SolarArray.hpp"
#include "Logger.hpp"
#include <cmath>

SolarArray::SolarArray(double efficiency, double base_input)
    : Subsystem("SolarArray"),
      bus_(nullptr),
      efficiency_(efficiency),
      base_input_(base_input),
      last_output_(0.0) {}

void SolarArray::initialize() {
    last_output_ = 0.0;
    // Initial wide row, include efficiency so header is stable
    Logger::instance().log_wide(
        "SolarArray", 0, 0.0,
        {"status","solar_input","output","efficiency"},
        {1.0, 0.0, 0.0, efficiency_}
    );
}

void SolarArray::tick(const TickContext& ctx) {
    // Simple diurnal-ish variation; keep your prior semantics:
    // base_input scaled by a smooth periodic function of time
    const double phase = 2.0 * M_PI * (ctx.time / (24.0 * 3600.0)); // daily cycle
    const double solar_input = base_input_ * (0.6 + 0.4 * std::max(0.0, std::sin(phase)));

    const double output = solar_input * efficiency_;
    last_output_ = output;

    if (bus_) bus_->addPower(output);

    Logger::instance().log_wide(
        "SolarArray", ctx.tick_index, ctx.time,
        {"status","solar_input","output","efficiency"},
        {1.0, solar_input, output, efficiency_}
    );
}

void SolarArray::shutdown() {
    // No -1 sentinel row (keeps file clean/consistent)
}

void SolarArray::setPowerBus(PowerBus* bus) {
    bus_ = bus;
}

double SolarArray::getLastOutput() const {
    return last_output_;
}
