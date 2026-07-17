#include "Radiator.hpp"

#include "Logger.hpp"
#include "PowerBus.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/*
    Radiator

    See Radiator.hpp for the model summary.

    Tick ordering contract

    BatteryThermal and CryoPanel tick before this node and deposit their
    (signed) heat via addHeatLoad. This node then integrates the loop
    temperature, moves the louvers, draws actuator power from the bus, and
    logs.
*/

// Global sunlight scale driven by OrbitModel in main.cpp (same pattern as
// SolarArray.cpp). 0.0 = full eclipse, 1.0 = full sun.
extern double g_orbit_solar_scale;

namespace {

// One shared column list keeps the header row (initialize) and the data rows
// (tick) from ever drifting apart.
const std::vector<std::string> kColumns = {
    "status",
    "T_loop_K",
    "louver_frac",
    "T_sink_K",
    "q_load_W",
    "q_reject_W",
    "actuator_req_W",
    "actuator_granted_W"
};

} // namespace

void Radiator::initialize() {
    loop_temp_K_           = 285.0;
    louver_frac_           = 0.5;
    heat_load_this_tick_W_ = 0.0;

    Logger::instance().log_wide(
        "Radiator", 0, 0.0, kColumns,
        {1.0, loop_temp_K_, louver_frac_, sink_night_K_, 0.0, 0.0, 0.0, 0.0}
    );
}

void Radiator::tick(const TickContext& ctx) {
    // Effective sink temperature rises when the panel sees sun and albedo.
    const double T_sink =
        SimHelpers::lerpDayNight(sink_night_K_, sink_day_K_,
                                 g_orbit_solar_scale);

    // Net heat arriving on the loop this tick: signed deposits from
    // BatteryThermal and CryoPanel plus the fixed avionics background.
    const double q_load_W = heat_load_this_tick_W_ + avionics_base_load_W_;

    // Louver proportional control around the loop setpoint.
    const double louver_prev = louver_frac_;
    louver_frac_ = SimHelpers::clamp01(
        0.5 + louver_gain_per_K_ * (loop_temp_K_ - loop_setpoint_K_));

    // Actuator draw: base pump/controller plus stroke cost for louver motion.
    const double actuator_req_W =
        pump_base_W_ + louver_stroke_W_ * std::fabs(louver_frac_ - louver_prev);

    double actuator_granted_W = 0.0;
    if (bus_) {
        actuator_granted_W = bus_->drawPower(actuator_req_W, ctx);
    }

    // Radiative rejection through the open louver fraction. T^4 written as
    // two squarings; std::pow is overkill for a constant integer exponent.
    const double T2_loop = loop_temp_K_ * loop_temp_K_;
    const double T2_sink = T_sink * T_sink;
    const double T4_loop = T2_loop * T2_loop;
    const double T4_sink = T2_sink * T2_sink;

    double q_reject_W =
        louver_frac_ * emissivity_ * SimHelpers::kStefanBoltzmannWm2K4 *
        panel_area_m2_ * (T4_loop - T4_sink);

    // The panel cannot absorb net heat from the sink through closed louvers;
    // clamp rejection at zero when the sink is hotter than the loop.
    if (q_reject_W < 0.0) {
        q_reject_W = 0.0;
    }

    // Pump work ends up in the coolant as heat.
    const double p_pump_heat_W = actuator_granted_W;

    // First-order explicit Euler integration of the loop temperature.
    const double net_W = q_load_W + p_pump_heat_W - q_reject_W;
    loop_temp_K_ += (net_W * ctx.dt) / c_loop_J_per_K_;

    loop_temp_K_ = SimHelpers::clampFiniteOrDefault(
        loop_temp_K_, 120.0, 450.0, loop_setpoint_K_);

    Logger::instance().log_wide(
        "Radiator", ctx.tick_index, ctx.time, kColumns,
        {
            1.0,
            loop_temp_K_,
            louver_frac_,
            T_sink,
            q_load_W,
            q_reject_W,
            actuator_req_W,
            actuator_granted_W
        }
    );

    // Consume the per-tick heat accumulator.
    heat_load_this_tick_W_ = 0.0;
}

void Radiator::shutdown() {}
