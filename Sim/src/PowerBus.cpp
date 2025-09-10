#include "PowerBus.hpp"
#include "Logger.hpp"
#include <algorithm>

PowerBus::PowerBus() : Subsystem("PowerBus") {}

void PowerBus::initialize() {
    available_power_      = 0.0;  // persists across ticks
    added_this_tick_      = 0.0;
    requested_this_tick_  = 0.0;
    granted_this_tick_    = 0.0;

    // Optional "came online" row. Safe to keep; remove if you don't want t=0.
    Logger::instance().log_wide("PowerBus", /*tick*/0, /*time*/0.0,
        {"status","available_added","total_draw_request","total_granted","remaining_power"},
        {1.0, 0.0, 0.0, 0.0, available_power_});
}

void PowerBus::tick(const TickContext& ctx) {
    // Summarize the tick in one wide row
    Logger::instance().log_wide("PowerBus", ctx.tick_index, ctx.time,
        {"status","available_added","total_draw_request","total_granted","remaining_power"},
        {1.0, added_this_tick_, requested_this_tick_, granted_this_tick_, available_power_});

    // Reset only per-tick counters; DO NOT zero available_power_.
    added_this_tick_     = 0.0;
    requested_this_tick_ = 0.0;
    granted_this_tick_   = 0.0;
}

void PowerBus::shutdown() {
    // status=0 final row
    Logger::instance().log_wide("PowerBus", -1, -1.0,
        {"status","available_added","total_draw_request","total_granted","remaining_power"},
        {0.0, 0.0, 0.0, 0.0, 0.0});
}

void PowerBus::addPower(double watts) {
    if (watts <= 0.0) return;
    available_power_ += watts;
    added_this_tick_ += watts;
}

double PowerBus::drawPower(double requested, const TickContext& /*ctx*/) {
    if (requested <= 0.0) return 0.0;

    // Partial-grant policy
    const double granted = std::min(requested, available_power_);
    available_power_    -= granted;

    requested_this_tick_ += requested;
    granted_this_tick_   += granted;
    return granted;
}

double PowerBus::getAvailablePower() const {
    return available_power_;
}
