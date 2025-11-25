#pragma once
#include "Subsystem.hpp"

// Forward declaration instead of including PowerBus.hpp
class PowerBus;

class Battery : public Subsystem {
public:
    explicit Battery(double capacity = 1000.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);

    // True energy storage (Wh)
    double getCharge() const;

    // Called by PowerBus when the bus cannot satisfy a load
    double discharge(double needed_W, double dt);

    // Called by PowerBus to store surplus bus energy
    void chargeFromSurplus(double surplus_W, double dt);

private:
    PowerBus* bus_;
    double capacity_;      // battery capacity (Wh)
    double charge_;        // current stored energy (Wh)

    double max_charge_rate_W_    = 200.0;   // limit charging speed (W)
    double max_discharge_rate_W_ = 2000.0;  // battery can output up to 2 kW
};
