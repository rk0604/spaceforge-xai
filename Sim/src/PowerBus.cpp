// Sim/src/PowerBus.cpp
#include "PowerBus.hpp"
#include "Logger.hpp"
#include <algorithm>

PowerBus::PowerBus() : Subsystem("PowerBus") {}

void PowerBus::setBattery(Battery* batt) {
    battery_ = batt;
}

void PowerBus::initialize() {
    available_power_              = 0.0;
    added_this_tick_              = 0.0;
    requested_this_tick_          = 0.0;
    granted_this_tick_            = 0.0;
    battery_discharged_this_tick_ = 0.0; // Initialize our new tracker

    // FIX: Match the new column names and initial values
    Logger::instance().log_wide(
        "PowerBus",
        0,
        0.0,
        {"status","solar_added","requested","granted","batt_drawn"},
        {1.0, 0.0, 0.0, 0.0, 0.0}
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

    // Bookkeep what was asked for this tick
    requested_this_tick_ += requested;

    // 1. Serve from bus first (e.g., direct solar generation)
    const double granted_from_bus = std::min(requested, available_power_);
    available_power_ -= granted_from_bus;

    const double remaining_need = requested - granted_from_bus;

    // 2. Pull remainder from battery, strictly clamped to physical limits
    double from_batt = 0.0;
    if (battery_ && remaining_need > 0.0) {
        
        // Query the battery's physical hardware limit
        // (Note: Adjust getMaxDischargeW() if your Battery.hpp uses a different getter name)
        const double max_batt_rate_W = battery_->getMaxDischargeW(); 
        
        // Calculate how much more power we are ALLOWED to pull from the battery this tick
        const double allowed_from_batt_W = std::max(0.0, max_batt_rate_W - battery_discharged_this_tick_);

        // Clamp the request so we don't blow past the 4000 W limit
        const double clamped_batt_request = std::min(remaining_need, allowed_from_batt_W);

        // Only attempt to discharge if we still have allowable rate limits
        if (clamped_batt_request > 0.0) {
            from_batt = battery_->discharge(clamped_batt_request, ctx.dt);
            battery_discharged_this_tick_ += from_batt;
        }
    }

    const double total_granted = granted_from_bus + from_batt;
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
        {"status","solar_added","requested","granted","batt_drawn"},
        {1.0, added_this_tick_, requested_this_tick_, granted_this_tick_, battery_discharged_this_tick_}
    );

    // Reset per-tick counters — bus does NOT store power across ticks
    available_power_              = 0.0;
    added_this_tick_              = 0.0;
    requested_this_tick_          = 0.0;
    granted_this_tick_            = 0.0;
    battery_discharged_this_tick_ = 0.0; // Reset our tracker for the next tick
}

void PowerBus::shutdown() {}

double PowerBus::getAvailablePower() const {
    return available_power_;
}

// Currently unused, but kept because it's declared in the header.
void PowerBus::logRow_(int /*tick*/, double /*time*/) {
    // no-op (logging happens in tick())
}