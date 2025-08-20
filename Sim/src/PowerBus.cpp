#include "PowerBus.hpp"
#include "Logger.hpp"

PowerBus::PowerBus() 
    : Subsystem("PowerBus"), available_power_(0.0) {}

void PowerBus::initialize() {
    available_power_ = 0.0;
    Logger::instance().log("PowerBus", 0, 0.0, {
        {"status", 1},
        {"available_power", available_power_}
    });
}

void PowerBus::tick(const TickContext& ctx) {
    // Log the available power at this tick
    Logger::instance().log("PowerBus", ctx.tick_index, ctx.time, {
        {"available_power", available_power_}
    });

    // Reset bus for the next tick
    available_power_ = 0.0;
}

void PowerBus::shutdown() {
    Logger::instance().log("PowerBus", -1, -1.0, {
        {"status", 0}
    });
}

void PowerBus::addPower(double watts) {
    available_power_ += watts;
}

double PowerBus::drawPower(double requested, const TickContext& ctx) {
    double granted = (requested <= available_power_) ? requested : available_power_;
    available_power_ -= granted;

    Logger::instance().log("PowerBus", ctx.tick_index, ctx.time, {
        {"draw_request", requested},
        {"granted", granted},
        {"remaining_power", available_power_}
    });

    return granted;
}

double PowerBus::getAvailablePower() const {
    return available_power_;
}
