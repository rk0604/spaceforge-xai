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
    // These return the CONFIGURED limits so regime columns stay constant.
    double getCapacityWh() const { return capacity_; }
    double getMaxChargeW() const { return max_charge_rate_W_; }
    double getMaxDischargeW() const { return max_discharge_rate_W_; }

    /*
        Effective limits after BatteryThermal derating. PowerBus clamps
        against these so a cold pack genuinely cannot deliver its full
        rated discharge power.
    */
    double getEffectiveMaxDischargeW() const {
        return max_discharge_rate_W_ * thermal_derate_dis_;
    }
    double getEffectiveMaxChargeW() const {
        return max_charge_rate_W_ * thermal_derate_chg_;
    }

    // Called by PowerBus when the bus cannot satisfy a load.
    double discharge(double needed_W, double dt);

    // Called by PowerBus to store surplus bus energy.
    void chargeFromSurplus(double surplus_W, double dt);

    /*
        Thermal derating pushed by BatteryThermal each tick.

        discharge_factor scales the effective max discharge power.
        charge_factor scales the effective max charge power.

        Both default to 1.0 so the battery behaves exactly as before when no
        BatteryThermal node is wired. Values are clamped to [0.1, 1.0] so a
        bad input can never fully disable the battery.
    */
    void setThermalDerating(double discharge_factor, double charge_factor) {
        if (std::isfinite(discharge_factor)) {
            thermal_derate_dis_ = std::clamp(discharge_factor, 0.1, 1.0);
        }
        if (std::isfinite(charge_factor)) {
            thermal_derate_chg_ = std::clamp(charge_factor, 0.1, 1.0);
        }
    }

    /*
        Previous-tick power flow snapshots used by BatteryThermal for Joule
        heating. Battery::tick runs last in the engine order, snapshots the
        per-tick accumulators, and resets them.
    */
    double getPrevTickDischargeW() const { return prev_tick_discharge_W_; }
    double getPrevTickChargeW() const { return prev_tick_charge_W_; }

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

    /*
        Thermal derating factors from BatteryThermal (1.0 = no derating).
    */
    double thermal_derate_dis_ = 1.0;
    double thermal_derate_chg_ = 1.0;

    /*
        Per-tick power flow accumulators and their previous-tick snapshots.

        The accumulators grow inside discharge() and chargeFromSurplus()
        during a tick; Battery::tick (which runs after PowerBus bookkeeping)
        snapshots and resets them.
    */
    double discharge_this_tick_W_ = 0.0;
    double charge_this_tick_W_    = 0.0;
    double prev_tick_discharge_W_ = 0.0;
    double prev_tick_charge_W_    = 0.0;
};