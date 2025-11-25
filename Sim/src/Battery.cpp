#include "Battery.hpp"
#include "Logger.hpp"
#include <algorithm>

Battery::Battery(double capacity)
    : Subsystem("Battery"),
      bus_(nullptr),
      capacity_(capacity),
      charge_(capacity / 2.0) {}   // Start 50% full

void Battery::initialize() {
    Logger::instance().log_wide(
        "Battery",
        0,
        0.0,
        {"status","charge_Wh","capacity_Wh"},
        {1.0, charge_, capacity_}
    );
}

void Battery::setPowerBus(PowerBus* bus) { bus_ = bus; }

double Battery::getCharge() const { return charge_; }

//
// Convert W → Wh over dt, respecting capacity
//
void Battery::chargeFromSurplus(double surplus_W, double dt) {
    if (surplus_W <= 0.0) return;

    // Apply charge rate limit
    double actual_W = std::min(surplus_W, max_charge_rate_W_);

    // Convert W to Wh
    double added_Wh = actual_W * (dt / 3600.0);

    charge_ = std::clamp(charge_ + added_Wh, 0.0, capacity_);
}

//
// Pull power from battery to support a load
// Returns W actually delivered
//
double Battery::discharge(double needed_W, double dt) {
    if (needed_W <= 0.0) return 0.0;

    // Cannot draw more than our discharge limit
    double deliverable_W = std::min(needed_W, max_discharge_rate_W_);

    // Convert battery Wh to max available power
    double max_possible_W = (charge_ * 3600.0) / dt;  // Wh → W

    double W_out = std::min(deliverable_W, max_possible_W);

    // Convert W back to Wh and subtract
    double used_Wh = W_out * (dt / 3600.0);
    charge_ = std::clamp(charge_ - used_Wh, 0.0, capacity_);

    return W_out;
}

void Battery::tick(const TickContext& ctx) {
    // Only logs status — bus controls all charging/discharging
    Logger::instance().log_wide(
        "Battery",
        ctx.tick_index,
        ctx.time,
        {"status","charge_Wh","capacity_Wh"},
        {1.0, charge_, capacity_}
    );
}

void Battery::shutdown() {}
