#pragma once

#include <algorithm>
#include <cmath>

#include "Subsystem.hpp"

// Forward declaration instead of including PowerBus.hpp
class PowerBus;

class Battery : public Subsystem {
public:
    /*
        Construct the battery subsystem.

        The default capacity is the Config 1 fallback value from the analytics
        tracker. Runtime experiments should still pass explicit values through
        the command line configuration path.
    */
    explicit Battery(double capacity = 6000.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);

    /*
        Configure the battery from runtime experiment parameters.

        This method is called from main.cpp after construction and before the
        simulation engine is initialized.

        capacity_wh
            Total battery energy capacity in watt hours.

        start_charge_wh
            Initial stored battery energy in watt hours.

        max_discharge_w
            Maximum battery output power available to the power bus.

        max_charge_w
            Maximum surplus power that can be stored by the battery.
    */
    void configure(double capacity_wh,
                   double start_charge_wh,
                   double max_discharge_w,
                   double max_charge_w) {
        if (std::isfinite(capacity_wh) && capacity_wh > 0.0) {
            capacity_ = capacity_wh;
        }

        if (std::isfinite(start_charge_wh) && start_charge_wh >= 0.0) {
            charge_ = std::clamp(start_charge_wh, 0.0, capacity_);
        }

        if (std::isfinite(max_discharge_w) && max_discharge_w >= 0.0) {
            max_discharge_rate_W_ = max_discharge_w;
        }

        if (std::isfinite(max_charge_w) && max_charge_w >= 0.0) {
            max_charge_rate_W_ = max_charge_w;
        }
    }

    // True energy storage in watt hours.
    double getCharge() const;

    // Regime getters used for logging and dataset conditioning.
    double getCapacityWh() const { return capacity_; }
    double getMaxChargeW() const { return max_charge_rate_W_; }
    double getMaxDischargeW() const { return max_discharge_rate_W_; }

    // Called by PowerBus when the bus cannot satisfy a load.
    double discharge(double needed_W, double dt);

    // Called by PowerBus to store surplus bus energy.
    void chargeFromSurplus(double surplus_W, double dt);

private:
    PowerBus* bus_;

    // Battery capacity in watt hours.
    double capacity_ = 6000.0;

    // Current stored energy in watt hours.
    double charge_ = 3000.0;

    /*
        Maximum battery charge rate in watts.

        Config 1 fallback value:
            3500 W
    */
    double max_charge_rate_W_ = 3500.0;

    /*
        Maximum battery discharge rate in watts.

        Config 1 fallback value:
            4000 W
    */
    double max_discharge_rate_W_ = 4000.0;
};