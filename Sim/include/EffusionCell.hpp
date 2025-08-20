#pragma once
#include "Subsystem.hpp"
#include "TickContext.hpp"

class EffusionCell : public Subsystem {
public:
    EffusionCell() : Subsystem("EffusionCell"), temperature_(300.0) {}
    void initialize() override {}
    void tick(const TickContext& ctx) override {}
    void shutdown() override {}

    // Called by HeaterBank
    void applyHeat(double watts, double dt);

    double getTemperature() const { return temperature_; }

private:
    double temperature_;
};
