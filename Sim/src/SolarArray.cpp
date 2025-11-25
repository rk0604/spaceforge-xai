#include "SolarArray.hpp"
#include "Logger.hpp"
#include <cmath>

// Global sunlight scale driven by OrbitModel in main.cpp.
// 0.0 = full eclipse, 1.0 = full sun, can be extended later.
extern double g_orbit_solar_scale;

SolarArray::SolarArray(double efficiency, double base_input)
    : Subsystem("SolarArray"),
      bus_(nullptr),
      efficiency_(efficiency),
      base_input_(base_input),
      last_output_(0.0) {}

void SolarArray::initialize() {
    last_output_ = 0.0;
    // Initial wide row, include efficiency so header is stable.
    // We also expose solar_scale explicitly for debugging.
    Logger::instance().log_wide(
        "SolarArray", 0, 0.0,
        {"status","solar_scale","solar_input","output","efficiency"},
        {1.0, 0.0, 0.0, 0.0, efficiency_}
    );
}

void SolarArray::tick(const TickContext& ctx) {
    // Use the same sunlight scale as the orbit model / CUP_BASE_SCALE.
    // In wake/dual/legacy mode, g_orbit_solar_scale is updated each tick
    // by main.cpp based on OrbitModel::state().solar_scale.
    // In power-only mode (no SPARTA / no OrbitModel), it stays at 1.0,
    // so the array produces a constant base_input_ * efficiency_.
    double solar_scale = g_orbit_solar_scale;

    if (!std::isfinite(solar_scale)) {
        solar_scale = 0.0;
    }
    if (solar_scale < 0.0) {
        solar_scale = 0.0;
    }
    if (solar_scale > 1.0) {
        solar_scale = 1.0;
    }

    const double solar_input = base_input_ * solar_scale;
    const double output      = solar_input * efficiency_;

    last_output_ = output;

    if (bus_) {
        bus_->addPower(output);
    }

    Logger::instance().log_wide(
        "SolarArray", ctx.tick_index, ctx.time,
        {"status","solar_scale","solar_input","output","efficiency"},
        {1.0, solar_scale, solar_input, output, efficiency_}
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
