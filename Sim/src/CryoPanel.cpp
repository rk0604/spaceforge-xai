#include "CryoPanel.hpp"

#include "Logger.hpp"
#include "PowerBus.hpp"
#include "Radiator.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/*
    CryoPanel

    See CryoPanel.hpp for the model summary.

    Tick ordering contract

    Ticks after the heater loads (so process state for this tick is already
    set by the scheduler) and before the Radiator, so rejected compressor
    heat lands on the coolant loop within the same tick.
*/

// Global sunlight scale driven by OrbitModel in main.cpp (same pattern as
// SolarArray.cpp). 0.0 = full eclipse, 1.0 = full sun.
extern double g_orbit_solar_scale;

namespace {

// One shared column list keeps the header row (initialize) and the data rows
// (tick) from ever drifting apart.
const std::vector<std::string> kColumns = {
    "status",
    "mode_regen",
    "T_cold_K",
    "duty",
    "power_req_W",
    "power_granted_W",
    "q_lift_W",
    "q_parasitic_W",
    "adsorbed_g",
    "ads_rate_g_min",
    "regen_ticks_left",
    "heat_to_radiator_W"
};

} // namespace

void CryoPanel::initialize() {
    cold_temp_K_      = 300.0;
    adsorbed_g_       = 0.0;
    regen_ticks_left_ = 0;

    Logger::instance().log_wide(
        "CryoPanel", 0, 0.0, kColumns,
        {1.0, 0.0, cold_temp_K_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    );
}

void CryoPanel::tick(const TickContext& ctx) {
    const double dt_min = ctx.dt / 60.0;

    // Effective environment temperature seen by the cold head.
    const double t_env_K =
        SimHelpers::lerpDayNight(env_night_K_, env_day_K_,
                                 g_orbit_solar_scale);

    // Parasitic heat leaking into the cold head. Positive warms the head.
    const double q_par_W = h_par_W_per_K_ * (t_env_K - cold_temp_K_);

    const bool in_regen = (regen_ticks_left_ > 0);

    double power_req_W       = 0.0;
    double power_granted_W   = 0.0;
    double q_lift_W          = 0.0;
    double duty              = 0.0;
    double ads_rate_g_min    = 0.0;
    double heat_to_rad_W     = 0.0;

    if (!in_regen) {
        // ---- COOLING mode ----

        // Throttle the compressor near the setpoint band.
        duty = SimHelpers::clamp01((cold_temp_K_ - t_duty_lo_K_) /
                                   (t_duty_hi_K_ - t_duty_lo_K_));

        power_req_W = p_standby_W_ + (p_full_W_ - p_standby_W_) * duty;

        if (bus_) {
            power_granted_W = bus_->drawPower(power_req_W, ctx);
        }

        // Thermal lift scales with the electrical power actually granted.
        q_lift_W = q_lift_max_W_ * (power_granted_W / p_full_W_);

        // Integrate the cold head temperature (explicit Euler).
        cold_temp_K_ += ((q_par_W - q_lift_W) * ctx.dt) / c_cold_J_per_K_;

        // Adsorption only once the panel is cold enough to capture.
        if (cold_temp_K_ < t_capture_K_) {
            ads_rate_g_min = r_base_g_per_min_;
            if (beam_on_ && process_flux_cm2s_ > 0.0) {
                ads_rate_g_min +=
                    r_flux_g_per_min_at_1e13_ * (process_flux_cm2s_ / 1.0e13);
            }
            adsorbed_g_ += ads_rate_g_min * dt_min;
        }

        // Everything the compressor ate plus the lifted heat is rejected.
        heat_to_rad_W = power_granted_W + q_lift_W;

        // Trigger regeneration when the panel is saturated.
        if (adsorbed_g_ >= m_regen_threshold_g_) {
            regen_ticks_left_ = regen_duration_ticks_;
        }
    } else {
        // ---- REGEN mode: cryocooler off, bake the panel ----

        power_req_W = p_regen_W_;

        if (bus_) {
            power_granted_W = bus_->drawPower(power_req_W, ctx);
        }

        // A fraction of the regen heat warms the cold head; the rest goes
        // to the radiator loop.
        const double q_regen_to_head_W = f_regen_ * power_granted_W;
        heat_to_rad_W = power_granted_W - q_regen_to_head_W;

        cold_temp_K_ +=
            ((q_par_W + q_regen_to_head_W) * ctx.dt) / c_cold_J_per_K_;

        // Desorb linearly toward the residual mass across the regen window.
        adsorbed_g_ =
            std::max(m_residual_g_, adsorbed_g_ - regen_release_g_per_tick_);

        regen_ticks_left_ -= 1;
    }

    cold_temp_K_ =
        SimHelpers::clampFiniteOrDefault(cold_temp_K_, 40.0, 400.0, t_env_K);

    if (!std::isfinite(adsorbed_g_) || adsorbed_g_ < 0.0) {
        adsorbed_g_ = 0.0;
    }

    if (radiator_ && heat_to_rad_W > 0.0) {
        radiator_->addHeatLoad(heat_to_rad_W);
    }

    Logger::instance().log_wide(
        "CryoPanel", ctx.tick_index, ctx.time, kColumns,
        {
            1.0,
            in_regen ? 1.0 : 0.0,
            cold_temp_K_,
            duty,
            power_req_W,
            power_granted_W,
            q_lift_W,
            q_par_W,
            adsorbed_g_,
            ads_rate_g_min,
            static_cast<double>(regen_ticks_left_),
            heat_to_rad_W
        }
    );
}

void CryoPanel::shutdown() {}
