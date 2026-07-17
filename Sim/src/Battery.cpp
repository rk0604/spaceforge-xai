#include "Battery.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <cmath>

/*
    Battery

    This subsystem stores and releases electrical energy for the power bus.

    Runtime configuration

    The constructor provides safe Config 1 fallback behavior.

    The configure method lives inline in Battery.hpp because main.cpp only
    needs a lightweight setter before engine initialization.

    Energy convention

    charge_ is stored in watt hours.

    Power requests are in watts.

    Time step values are in seconds.

    Conversion

    Wh = W * seconds / 3600
*/

Battery::Battery(double capacity)
    : Subsystem("Battery"),
      bus_(nullptr),
      capacity_((std::isfinite(capacity) && capacity > 0.0) ? capacity : 6000.0),
      charge_(capacity_ / 2.0) {
    /*
        Start at half charge by default.

        For dataset generation, main.cpp should immediately override this
        through battery.configure using values loaded from the tracker.
    */
}

void Battery::initialize() {
    /*
        Emit the initial battery state.

        This row is useful for confirming that runtime command line values were
        applied before the simulation engine started ticking.
    */
    Logger::instance().log_wide(
        "Battery",
        0,
        0.0,
        {
            "status",
            "charge_Wh",
            "capacity_Wh",
            "max_charge_W",
            "max_discharge_W"
        },
        {
            1.0,
            charge_,
            capacity_,
            max_charge_rate_W_,
            max_discharge_rate_W_
        }
    );
}

void Battery::setPowerBus(PowerBus* bus) {
    /*
        Store the bus pointer so the battery can participate in the power
        system without owning the bus.
    */
    bus_ = bus;
}

double Battery::getCharge() const {
    /*
        Return current stored battery energy in watt hours.
    */
    return charge_;
}

void Battery::chargeFromSurplus(double surplus_W, double dt) {
    /*
        Store surplus power from the power bus.

        Invalid or nonpositive inputs are ignored so transient bad values do
        not corrupt the battery state.
    */
    if (!std::isfinite(surplus_W) || surplus_W <= 0.0) {
        return;
    }

    if (!std::isfinite(dt) || dt <= 0.0) {
        return;
    }

    /*
        Apply the charge rate limit first, including any thermal derating
        pushed by BatteryThermal.

        The power bus may have more surplus than the battery can safely accept.
    */
    const double actual_W = std::min(surplus_W, getEffectiveMaxChargeW());

    /*
        Convert delivered charging power into stored energy.
    */
    const double added_Wh = actual_W * (dt / 3600.0);

    /*
        Clamp to the configured physical capacity.
    */
    charge_ = std::clamp(charge_ + added_Wh, 0.0, capacity_);

    // Track the accepted charging power for BatteryThermal Joule heating.
    charge_this_tick_W_ += actual_W;
}

double Battery::discharge(double needed_W, double dt) {
    /*
        Provide battery power when the bus cannot satisfy a load.

        Return value is the actual output power in watts.
    */
    if (!std::isfinite(needed_W) || needed_W <= 0.0) {
        return 0.0;
    }

    if (!std::isfinite(dt) || dt <= 0.0) {
        return 0.0;
    }

    /*
        The battery cannot exceed the discharge power limit, including any
        thermal derating pushed by BatteryThermal.
    */
    const double rate_limited_W = std::min(needed_W, getEffectiveMaxDischargeW());

    /*
        The battery also cannot deliver more energy than it currently stores.

        charge_ is watt hours.

        Multiplying by 3600 and dividing by dt gives the maximum possible
        average power over this tick.
    */
    const double energy_limited_W = (charge_ * 3600.0) / dt;

    /*
        Actual output power is limited by both power electronics and stored
        energy.
    */
    const double output_W = std::min(rate_limited_W, energy_limited_W);

    /*
        Convert delivered power back into watt hours and remove it from the
        stored charge.
    */
    const double used_Wh = output_W * (dt / 3600.0);

    charge_ = std::clamp(charge_ - used_Wh, 0.0, capacity_);

    // Track the delivered discharge power for BatteryThermal Joule heating.
    discharge_this_tick_W_ += output_W;

    return output_W;
}

void Battery::tick(const TickContext& ctx) {
    /*
        The power bus controls charge and discharge.

        The battery tick only logs the state after bus accounting has updated
        the stored charge for this tick.

        Battery ticks last in the engine order, so both discharge (loads)
        and charge (bus surplus) flows for this tick are final here. Snapshot
        them for BatteryThermal, which consumes them next tick with a
        deliberate one-tick lag, then reset the accumulators.
    */
    prev_tick_discharge_W_ = discharge_this_tick_W_;
    prev_tick_charge_W_    = charge_this_tick_W_;
    discharge_this_tick_W_ = 0.0;
    charge_this_tick_W_    = 0.0;

    Logger::instance().log_wide(
        "Battery",
        ctx.tick_index,
        ctx.time,
        {
            "status",
            "charge_Wh",
            "capacity_Wh",
            "max_charge_W",
            "max_discharge_W"
        },
        {
            1.0,
            charge_,
            capacity_,
            max_charge_rate_W_,
            max_discharge_rate_W_
        }
    );
}

void Battery::shutdown() {
    /*
        No owned external resources require shutdown.
    */
}