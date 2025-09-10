#include "Battery.hpp"
#include "Logger.hpp"
#include <algorithm>

Battery::Battery(double capacity)
    : Subsystem("Battery"),
      bus_(nullptr),
      capacity_(capacity),
      charge_(capacity / 2.0) {}   // start ~50% as before

void Battery::initialize() {
    // Initial wide row with current charge; no extra per-key rows
    Logger::instance().log_wide(
        "Battery", 0, 0.0,
        {"status","required","drawn","deficit","low_flag","charge"},
        {1.0, 0.0, 0.0, 0.0, (charge_ < 0.2*capacity_) ? 1.0 : 0.0, charge_}
    );
}

void Battery::tick(const TickContext& ctx) {
    const double required = request_per_tick_;

    double drawn = 0.0;
    if (bus_) drawn = bus_->drawPower(required, ctx);

    const double deficit = std::max(0.0, required - drawn);

    // Update state (simple charger: what we draw becomes stored energy)
    charge_ = std::clamp(charge_ + drawn, 0.0, capacity_);

    const double low_flag = (charge_ < 0.2 * capacity_) ? 1.0 : 0.0;

    // One wide row for this tick
    Logger::instance().log_wide(
        "Battery", ctx.tick_index, ctx.time,
        {"status","required","drawn","deficit","low_flag","charge"},
        {1.0, required, drawn, deficit, low_flag, charge_}
    );
}

void Battery::shutdown() {
    // No -1 sentinel row (keeps file clean/consistent)
}

void Battery::setPowerBus(PowerBus* bus) {
    bus_ = bus;
}

double Battery::getCharge() const {
    return charge_;
}
