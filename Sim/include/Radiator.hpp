#pragma once

#include <algorithm>
#include <cmath>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class PowerBus;

/*
    Radiator

    Louvered body-mounted radiator panel plus pumped coolant loop node.

    Other hardware nodes (BatteryThermal, CryoPanel) reject their waste heat
    into this node during their tick via addHeatLoad. The radiator integrates
    the loop temperature and rejects heat to space through a louver-modulated
    Stefan-Boltzmann law with an orbit-dependent effective sink temperature.

    Model summary

        T_sink_eff = T_sink_night + (T_sink_day - T_sink_night) * solar_scale

        Q_reject = louver_frac * eps * sigma * A * (T_loop^4 - T_sink_eff^4)

        dT_loop / dt = (Q_load + P_pump_heat - Q_reject) / C_loop

    Louver control is proportional around the loop setpoint:

        louver_frac = clamp01(0.5 + k_louver * (T_loop - T_set))

    The louver actuator draws electrical power proportional to how far the
    louvers moved this tick, plus a small constant pump/controller draw:

        P_req = P_base + P_stroke * |louver_frac - louver_frac_prev|

    All electrical draw is requested from the shared PowerBus, so the
    radiator competes with the heaters for the same watts.
*/
class Radiator : public Subsystem {
public:
    Radiator() : Subsystem("Radiator") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus) { bus_ = bus; }

    /*
        Deposit heat (watts, signed) into the coolant loop for the current
        tick. Positive adds heat to the loop; negative removes it (e.g. the
        loop conductively warming a colder battery pack), which keeps energy
        conserved between coupled nodes.

        Callers are other subsystems that tick before the radiator inside the
        same engine tick. The accumulator is consumed and reset when the
        radiator itself ticks.
    */
    void addHeatLoad(double watts) {
        if (std::isfinite(watts)) {
            heat_load_this_tick_W_ += watts;
        }
    }

    // Current coolant loop temperature, used by BatteryThermal as its sink.
    double getLoopTempK() const { return loop_temp_K_; }

private:
    PowerBus* bus_ = nullptr;

    // Coolant loop temperature in kelvin.
    double loop_temp_K_{285.0};

    // Louver open fraction, 0 = fully closed, 1 = fully open.
    double louver_frac_{0.5};

    // Waste heat deposited by other nodes during the current tick (watts).
    double heat_load_this_tick_W_{0.0};

    // ---- Model constants ----

    // Lumped loop + panel thermal capacitance (J per K).
    double c_loop_J_per_K_{25000.0};

    // Panel radiating area (m^2) and IR emissivity.
    double panel_area_m2_{3.0};
    double emissivity_{0.85};

    // Effective sink temperature seen by the panel (K).
    double sink_night_K_{100.0};
    double sink_day_K_{230.0};

    // Louver proportional control around the loop setpoint.
    double loop_setpoint_K_{285.0};
    double louver_gain_per_K_{0.1};

    // Electrical draw: constant pump/controller base plus stroke cost.
    double pump_base_W_{15.0};
    double louver_stroke_W_{25.0};

    // Fixed avionics waste heat always present on the loop (watts).
    double avionics_base_load_W_{120.0};
};
