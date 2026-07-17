#pragma once

#include <algorithm>
#include <cmath>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class Battery;
class PowerBus;
class Radiator;

/*
    BatteryThermal

    Battery pack cell-temperature node with survival heaters.

    The pack self-heats from I^2 R losses whenever it charges or discharges,
    cools conductively into the radiator coolant loop, and carries real
    survival heaters that draw bus power during cold eclipse passes.

    Model summary

        I_batt   = (P_discharge_prev + P_charge_prev) / V_bus_nom

        Q_joule  = I_batt^2 * R_internal

        Q_cool   = h_batt * (T_batt - T_loop)

        dT_batt / dt = (Q_joule + P_survival_granted - Q_cool) / C_batt

    Power flow inputs use the previous tick's battery charge and discharge
    powers (snapshotted by Battery::tick), because bus surplus charging for
    the current tick is only finalized after this node has already run. The
    one-tick lag is physical and stable at dt = 60 s.

    Survival heater is bang-bang with hysteresis:

        ON  when T_batt <  T_survival_on
        OFF when T_batt >= T_survival_off

    Feedback into the electrical model: cell temperature derates the
    battery's effective discharge and charge limits through
    Battery::setThermalDerating. Cold packs cannot push their full rated
    discharge power; hot packs must taper charging.

        derate_dis = clamp(0.5 + 0.5 * (T - T_dis_lo) / (T_dis_hi - T_dis_lo), 0.5, 1.0)
        derate_chg = clamp(1.0 - 0.6 * (T - T_chg_lo) / (T_chg_hi - T_chg_lo), 0.4, 1.0)
*/
class BatteryThermal : public Subsystem {
public:
    BatteryThermal() : Subsystem("BatteryThermal") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setBattery(Battery* batt)   { battery_  = batt; }
    void setPowerBus(PowerBus* bus)  { bus_      = bus; }
    void setRadiator(Radiator* rad)  { radiator_ = rad; }

private:
    Battery*  battery_  = nullptr;
    PowerBus* bus_      = nullptr;
    Radiator* radiator_ = nullptr;

    // Pack temperature in kelvin. Starts at room temperature.
    double pack_temp_K_{293.0};

    // Survival heater latch state for the bang-bang controller.
    bool survival_on_{false};

    // ---- Model constants ----

    // Lumped pack thermal capacitance (J per K). Roughly a 40 kg pack.
    double c_pack_J_per_K_{40000.0};

    // Conductive coupling from pack to the radiator coolant loop (W per K).
    double h_pack_W_per_K_{5.0};

    // Fallback sink temperature when no radiator is wired (K).
    double fallback_sink_K_{285.0};

    // Electrical model for Joule heating.
    double v_bus_nom_V_{120.0};
    double r_internal_ohm_{0.05};

    // Survival heater bang-bang thresholds and power.
    double survival_on_K_{275.0};
    double survival_off_K_{278.0};
    double survival_power_W_{150.0};

    // Discharge derating ramp: 0.5 at T_dis_lo, 1.0 at T_dis_hi.
    double t_dis_lo_K_{263.0};
    double t_dis_hi_K_{288.0};

    // Charge derating ramp: 1.0 at T_chg_lo, 0.4 at T_chg_hi.
    double t_chg_lo_K_{303.0};
    double t_chg_hi_K_{323.0};
};
