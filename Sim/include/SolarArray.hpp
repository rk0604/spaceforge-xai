#pragma once
#include "Subsystem.hpp"
#include "PowerBus.hpp"

class SolarArray : public Subsystem {
public:
    explicit SolarArray(double efficiency = 0.2, double base_input = 1000.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);
    double getLastOutput() const;

private:
    PowerBus* bus_;
    double efficiency_;   // fraction converted to power
    double base_input_;   // baseline solar input (W)
    double last_output_;  // last tick output (W)
};
