#pragma once
#include "Subsystem.hpp"
#include "TickContext.hpp"
#include "Battery.hpp"

class PowerBus : public Subsystem {
public:
    PowerBus();

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    // Add generation during this tick
    void addPower(double watts);

    // Consumer power request
    double drawPower(double requested, const TickContext& ctx);

    // Called by main to link the battery
    void setBattery(Battery* batt);

    double getAvailablePower() const;

private:
    void logRow_(int tick, double time);

    double available_power_{0.0};

    double added_this_tick_{0.0};
    double requested_this_tick_{0.0};
    double granted_this_tick_{0.0};

    Battery* battery_ = nullptr;
};
