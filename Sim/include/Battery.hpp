#pragma once
#include "Subsystem.hpp"

// Forward declaration instead of including PowerBus.hpp
class PowerBus;

class Battery : public Subsystem {
public:
    explicit Battery(double capacity = 6000.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);

    // True energy storage (Wh)
    double getCharge() const;

    // Regime getters (for logging + dataset conditioning)
    double getCapacityWh() const { return capacity_; }
    double getMaxChargeW() const { return max_charge_rate_W_; }
    double getMaxDischargeW() const { return max_discharge_rate_W_; }

    // Called by PowerBus when the bus cannot satisfy a load
    double discharge(double needed_W, double dt);

    // Called by PowerBus to store surplus bus energy
    void chargeFromSurplus(double surplus_W, double dt);

private:
    PowerBus* bus_;
    double capacity_;      // battery capacity (Wh)
    double charge_;        // current stored energy (Wh)

    // Raised from 200 W -> 800 W so the battery can actually bank solar surplus.
    // This should improve job survival but still allow failures under bad timing/eclipses.
    double max_charge_rate_W_    = 1600.0;

    double max_discharge_rate_W_ = 1200.0;  // battery can output up to 2 kW
};
