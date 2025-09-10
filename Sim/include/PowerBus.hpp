#pragma once
#include "Subsystem.hpp"
#include "TickContext.hpp"
#include <string>

class PowerBus : public Subsystem {
public:
    PowerBus();

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    // Add generation during this tick
    void addPower(double watts);

    // Consumers request power; returns amount granted (≤ requested)
    double drawPower(double requested, const TickContext& ctx);

    double getAvailablePower() const;

private:
    // One snapshot row per tick via Logger::log_wide(...)
    void logRow_(int tick, double time, double status);

    // Bus state
    double available_power_{0.0};

    // Per-tick accumulators (reset in tick())
    double added_this_tick_{0.0};
    double requested_this_tick_{0.0};
    double granted_this_tick_{0.0};
};
