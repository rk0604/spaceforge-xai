#pragma once
#include "Subsystem.hpp"
#include "PowerBus.hpp"

class SolarArray : public Subsystem {
public:
    // Default: ~30 % efficient array, ~5.7 kW incident sunlight at full sun.
    // This yields about 1.7 kW electrical output when solar_scale = 1.0.
    explicit SolarArray(double efficiency = 0.30,
                        double base_input = 5667.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);

    double getLastOutput() const;

    // Regime getters (for logging + dataset conditioning)
    double getEfficiency() const { return efficiency_; }
    double getBaseInputW() const { return base_input_; }

private:
    PowerBus* bus_;
    double efficiency_;   // fraction of incident solar converted to electrical power
    double base_input_;   // baseline solar input at solar_scale=1 (W)
    double last_output_;  // last tick electrical output (W)
};
