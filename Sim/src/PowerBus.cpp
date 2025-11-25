#include "PowerBus.hpp"
#include "Logger.hpp"
#include <algorithm>

PowerBus::PowerBus() : Subsystem("PowerBus") {}

void PowerBus::setBattery(Battery* batt) {
    battery_ = batt;
}

void PowerBus::initialize() {
    available_power_      = 0.0;
    added_this_tick_      = 0.0;
    requested_this_tick_  = 0.0;
    granted_this_tick_    = 0.0;

    Logger::instance().log_wide(
        "PowerBus",
        0,
        0.0,
        {"status","available_added","requested","granted","remaining"},
        {1.0, 0.0, 0.0, 0.0, available_power_}
    );
}

void PowerBus::addPower(double watts) {
    if (watts > 0.0) {
        available_power_ += watts;
        added_this_tick_ += watts;
    }
}

double PowerBus::drawPower(double requested, const TickContext& ctx) {
    if (requested <= 0.0) return 0.0;

    requested_this_tick_ += requested;

    // First try to serve from bus
    double granted_from_bus = std::min(requested, available_power_);
    available_power_ -= granted_from_bus;

    double remaining_need = requested - granted_from_bus;

    double from_batt = 0.0;
    if (battery_ && remaining_need > 0.0) {
        from_batt = battery_->discharge(remaining_need, ctx.dt);
    }

    double total_granted = granted_from_bus + from_batt;
    granted_this_tick_ += total_granted;

    return total_granted;
}

void PowerBus::tick(const TickContext& ctx) {
    // After all loads have drawn, any leftover power goes to battery
    if (battery_ && available_power_ > 0.0) {
        battery_->chargeFromSurplus(available_power_, ctx.dt);
    }

    // Log this tick
    Logger::instance().log_wide(
        "PowerBus",
        ctx.tick_index,
        ctx.time,
        {"status","available_added","requested","granted","remaining"},
        {1.0, added_this_tick_, requested_this_tick_, granted_this_tick_, available_power_}
    );

    // Reset per-tick counters — bus does NOT store power
    available_power_      = 0.0;
    added_this_tick_      = 0.0;
    requested_this_tick_  = 0.0;
    granted_this_tick_    = 0.0;
}

void PowerBus::shutdown() {}

double PowerBus::getAvailablePower() const {
    return available_power_;
}
