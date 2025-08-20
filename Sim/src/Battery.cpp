#include "Battery.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <iostream>

Battery::Battery(double capacity)
    : Subsystem("Battery"),
      capacity_(capacity),
      charge_(capacity / 2.0),  // start at 50% charge by default
      bus_(nullptr) {}

void Battery::initialize() {
    Logger::instance().log("Battery", 0, 0.0, {
        {"status", 1},
        {"initial_charge", charge_}
    });
}

void Battery::tick(const TickContext& ctx) {
    if (!bus_) return;

    // Request a small trickle to keep battery topped up
    double required = std::min(5.0, capacity_ - charge_);
    double drawn = bus_->drawPower(required, ctx);

    charge_ += drawn;
    if (charge_ > capacity_) charge_ = capacity_;

    double deficit = required - drawn;

    Logger::instance().log("Battery", ctx.tick_index, ctx.time, {
        {"charge", charge_},
        {"required", required},
        {"drawn", drawn},
        {"deficit", deficit},
        {"low_flag", (charge_ < capacity_ * 0.2) ? 1 : 0}
    });
}

void Battery::shutdown() {
    Logger::instance().log("Battery", -1, -1.0, {
        {"status", 0}
    });
}

void Battery::setPowerBus(PowerBus* bus) {
    bus_ = bus;
}

double Battery::getCharge() const {
    return charge_;
}
