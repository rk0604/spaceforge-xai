// Sim/src/EffusionCell.cpp
#include "EffusionCell.hpp"
#include "Logger.hpp"
#include "WakeChamber.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// Avoid duplicate logs per process if tick() is called twice on one rank.
static int s_last_logged_tick = -1;

// Gate streaks are computed in main.cpp and mirrored here so they appear in
// EffusionCell.csv without changing the logger call sites there.
int g_underflux_streak_for_log = 0;
int g_temp_miss_streak_for_log = 0;

namespace {
constexpr double kIdleBaselineTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;
}

void EffusionCell::initialize() {
    last_heat_W_           = 0.0;
    heat_input_w_          = 0.0;
    last_p_loss_W_         = 0.0;
    last_net_W_            = 0.0;
    temperature_           = kIdleBaselineTempK;
    target_temp_K_         = temperature_;
    last_pushed_temp_      = temperature_;

    // Initialize the orbit-aware environment using the current solar scale.
    // main.cpp will update this each tick before source demand is computed.
    setOrbitThermalEnvironment(solar_scale_);
}

void EffusionCell::tick(const TickContext& ctx) {
    // De-duplicate logging on this process in case tick() is invoked twice.
    if (ctx.tick_index == s_last_logged_tick) {
        return;
    }
    s_last_logged_tick = ctx.tick_index;

    Logger::instance().log_wide(
        "EffusionCell",
        ctx.tick_index,
        ctx.time,
        {
            "status",
            "act_temp_K",
            "target_temp_K",
            "T_env_eff_K",
            "solar_scale",
            "P_solar_abs_W",
            "heatInput_w",
            "underflux_streak",
            "temp_miss_streak",
            "P_loss_W",
            "P_net_W",
            "C_J",
            "h_WK"
        },
        {
            1.0,
            temperature_,
            target_temp_K_,
            ambient_temp_K_,
            solar_scale_,
            solar_absorbed_power_W_,
            heat_input_w_,
            static_cast<double>(g_underflux_streak_for_log),
            static_cast<double>(g_temp_miss_streak_for_log),
            last_p_loss_W_,
            last_net_W_,
            c_j_per_k_,
            h_w_per_k_
        }
    );

    // Optionally push the current actual source temperature into SPARTA at a
    // limited cadence when the temperature has changed enough to matter.
    if (sparta_ctrl_
        && (ctx.tick_index % push_every_ticks_ == 0)
        && std::fabs(temperature_ - last_pushed_temp_) >= push_threshold_K_) {
        sparta_ctrl_->setParameter("cell_temp_K", temperature_);
        sparta_ctrl_->markDirtyReload();
        last_pushed_temp_ = temperature_;
    }

    // Reset the per-tick reported heat input after logging.
    heat_input_w_ = 0.0;
}

void EffusionCell::shutdown() {
    // No special shutdown rows are needed.
}

void EffusionCell::setOrbitThermalEnvironment(double solar_scale) {
    double s = solar_scale;
    if (!std::isfinite(s)) {
        s = 0.0;
    }
    s = std::clamp(s, 0.0, 1.0);

    solar_scale_ = s;

    // Effective environment temperature seen by the source for the current
    // orbital illumination state.
    ambient_temp_K_ =
        night_ambient_temp_K_
        + (day_ambient_temp_K_ - night_ambient_temp_K_) * solar_scale_;

    // Absorbed solar heating power acting directly on the source.
    solar_absorbed_power_W_ =
        solar_absorptivity_ * projected_area_m2_ * solar_constant_W_m2_ * solar_scale_;

    if (!std::isfinite(ambient_temp_K_)) {
        ambient_temp_K_ = kIdleBaselineTempK;
    }
    if (!std::isfinite(solar_absorbed_power_W_) || solar_absorbed_power_W_ < 0.0) {
        solar_absorbed_power_W_ = 0.0;
    }
}

void EffusionCell::applyHeat(double watts, double dt) {
    const double pin_W  = std::max(0.0, watts);
    const double dt_pos = std::max(0.0, dt);

    /*
        First-order RC thermal update with orbit-aware forcing.

        P_loss is linear with temperature difference to the current effective
        environment. Negative P_loss is allowed when the source is colder than
        the effective environment, which lets the environment passively warm
        the source.

        P_net = P_heater + P_solar_abs - P_loss
    */
    last_p_loss_W_ = h_w_per_k_ * (temperature_ - ambient_temp_K_);
    last_net_W_    = pin_W + solar_absorbed_power_W_ - last_p_loss_W_;

    const double dT_K = (last_net_W_ / c_j_per_k_) * dt_pos;
    temperature_ += dT_K;

    // Basic safety clamps.
    if (!std::isfinite(temperature_)) {
        temperature_ = ambient_temp_K_;
    }
    if (temperature_ < 0.0) {
        temperature_ = 0.0;
    }

    last_heat_W_  = pin_W;
    heat_input_w_ = pin_W;
}

void EffusionCell::setTargetTempK(double T_K) {
    // Clamp invalid or non-physical targets back to the idle baseline.
    if (!std::isfinite(T_K) || T_K < 0.0) {
        target_temp_K_ = kIdleBaselineTempK;
        return;
    }

    target_temp_K_ = T_K;
}

bool EffusionCell::hasMeaningfulTarget() const {
    if (!std::isfinite(target_temp_K_)) {
        return false;
    }

    return target_temp_K_ > (kIdleBaselineTempK + kMeaningfulTargetMarginK);
}

bool EffusionCell::isAtTarget(double readiness_fraction) const {
    // Preserve legacy one-sided readiness semantics for existing callers.
    if (!hasMeaningfulTarget()) {
        return true;
    }

    if (!std::isfinite(temperature_) || !std::isfinite(target_temp_K_)) {
        return false;
    }

    double frac = readiness_fraction;
    if (!std::isfinite(frac)) {
        frac = 0.90;
    }
    frac = std::clamp(frac, 0.0, 1.0);

    return temperature_ >= (frac * target_temp_K_);
}

EffusionCell::ThermalBandState
EffusionCell::getThermalBandState(double lower_readiness_fraction,
                                  double upper_readiness_fraction) const {
    // No meaningful target means the source is operationally idle and should
    // not block scheduler readiness.
    if (!hasMeaningfulTarget()) {
        return ThermalBandState::Idle;
    }

    if (!std::isfinite(temperature_) || !std::isfinite(target_temp_K_)) {
        // Conservative choice: invalid state is treated as below target.
        return ThermalBandState::BelowTargetBand;
    }

    double lower = lower_readiness_fraction;
    double upper = upper_readiness_fraction;

    if (!std::isfinite(lower)) {
        lower = 0.90;
    }
    if (!std::isfinite(upper)) {
        upper = 1.05;
    }

    lower = std::clamp(lower, 0.0, 1.0);
    upper = std::max(upper, lower);

    const double lower_bound_K = lower * target_temp_K_;
    const double upper_bound_K = upper * target_temp_K_;

    if (temperature_ < lower_bound_K) {
        return ThermalBandState::BelowTargetBand;
    }
    if (temperature_ > upper_bound_K) {
        return ThermalBandState::AboveTargetBand;
    }

    return ThermalBandState::WithinTargetBand;
}

bool EffusionCell::isBelowTargetBand(double lower_readiness_fraction) const {
    return getThermalBandState(lower_readiness_fraction,
                               std::max(1.05, lower_readiness_fraction))
           == ThermalBandState::BelowTargetBand;
}

bool EffusionCell::isWithinTargetBand(double lower_readiness_fraction,
                                      double upper_readiness_fraction) const {
    return getThermalBandState(lower_readiness_fraction, upper_readiness_fraction)
           == ThermalBandState::WithinTargetBand;
}

bool EffusionCell::isAboveTargetBand(double upper_readiness_fraction) const {
    double upper = upper_readiness_fraction;
    if (!std::isfinite(upper)) {
        upper = 1.05;
    }
    upper = std::max(upper, 0.90);

    return getThermalBandState(0.90, upper) == ThermalBandState::AboveTargetBand;
}