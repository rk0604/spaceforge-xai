#include "EffusionCell.hpp"
#include "Logger.hpp"
#include "WakeChamber.hpp"

#include <algorithm>
#include <cmath>

/*
    EffusionCell

    This file owns the actual source temperature evolution.

    Runtime configuration

    C_J, h_WK, night ambient, and day ambient are configured through inline
    setters in EffusionCell.hpp. Those setters are called from main.cpp before
    engine initialization.

    Thermal model

    T_env_eff = T_night + (T_day minus T_night) * solar_scale

    P_solar_abs = alpha_abs * A_proj * G_solar * solar_scale

    P_loss = h_WK * (T_cell minus T_env_eff)

    P_net = P_heater + P_solar_abs minus P_loss

    dT = P_net / C_J * dt

    Sign convention

    Positive P_loss means heat leaves the effusion cell.

    Negative P_loss means the effective environment warms the effusion cell.
*/

/*
    Avoid duplicate logs per process if tick is accidentally called more than
    once on the same rank for the same tick index.
*/
static int s_last_logged_tick = -1;

/*
    These streaks are computed in main.cpp because main.cpp owns the live
    deposition gate.

    They are mirrored here so EffusionCell.csv contains the same failure gate
    counters without moving scheduler logic into the effusion thermal model.
*/
int g_underflux_streak_for_log = 0;
int g_temp_miss_streak_for_log = 0;

namespace {

constexpr double kIdleBaselineTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;

}  // namespace

void EffusionCell::initialize() {
    /*
        Reset dynamic thermal state.

        The runtime configuration constants are not reset here. Values already
        loaded from command line arguments must remain active after
        initialization.
    */
    last_heat_W_      = 0.0;
    heat_input_w_     = 0.0;
    last_p_loss_W_    = 0.0;
    last_net_W_       = 0.0;
    temperature_      = kIdleBaselineTempK;
    target_temp_K_    = temperature_;
    last_pushed_temp_ = temperature_;

    /*
        Recompute the effective environment from the current configured ambient
        values and the current stored solar scale.
    */
    setOrbitThermalEnvironment(solar_scale_);
}

void EffusionCell::tick(const TickContext& ctx) {
    /*
        De duplicate logging for this process.
    */
    if (ctx.tick_index == s_last_logged_tick) {
        return;
    }

    s_last_logged_tick = ctx.tick_index;

    /*
        Emit source thermal state and runtime configuration values.

        C_J and h_WK are included so each job folder can prove which source
        thermal model was used.
    */
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
            "h_WK",
            "night_ambient_K",
            "day_ambient_K"
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
            h_w_per_k_,
            night_ambient_temp_K_,
            day_ambient_temp_K_
        }
    );

    /*
        Optionally push the actual source temperature into SPARTA.

        This remains rate limited so small thermal changes do not force
        unnecessary deck reloads.
    */
    if (sparta_ctrl_ &&
        (ctx.tick_index % push_every_ticks_ == 0) &&
        std::fabs(temperature_ - last_pushed_temp_) >= push_threshold_K_) {
        sparta_ctrl_->setParameter("cell_temp_K", temperature_);
        sparta_ctrl_->markDirtyReload();
        last_pushed_temp_ = temperature_;
    }

    /*
        heat_input_w_ is a per tick reporting value.

        Reset after logging so the next row only reflects heat applied during
        the next tick.
    */
    heat_input_w_ = 0.0;
}

void EffusionCell::shutdown() {
    /*
        No owned external resources require shutdown.
    */
}

void EffusionCell::setOrbitThermalEnvironment(double solar_scale) {
    /*
        Clamp orbit illumination into the expected physical range.
    */
    double s = solar_scale;
    if (!std::isfinite(s)) {
        s = 0.0;
    }

    solar_scale_ = std::clamp(s, 0.0, 1.0);

    /*
        Compute the effective ambient temperature seen by the source.

        This is the corrected model where eclipse uses the configured night
        ambient value, normally 250 K for the regenerated dataset.
    */
    ambient_temp_K_ =
        night_ambient_temp_K_ +
        (day_ambient_temp_K_ - night_ambient_temp_K_) * solar_scale_;

    /*
        Compute absorbed solar heating acting directly on the source node.
    */
    solar_absorbed_power_W_ =
        solar_absorptivity_ *
        projected_area_m2_ *
        solar_constant_W_m2_ *
        solar_scale_;

    /*
        Safety repair for invalid derived values.
    */
    if (!std::isfinite(ambient_temp_K_)) {
        ambient_temp_K_ = kIdleBaselineTempK;
    }

    if (!std::isfinite(solar_absorbed_power_W_) ||
        solar_absorbed_power_W_ < 0.0) {
        solar_absorbed_power_W_ = 0.0;
    }
}

void EffusionCell::applyHeat(double watts, double dt) {
    /*
        Sanitize delivered power and time step.
    */
    const double pin_W =
        (std::isfinite(watts) && watts > 0.0) ? watts : 0.0;

    const double dt_pos =
        (std::isfinite(dt) && dt > 0.0) ? dt : 0.0;

    /*
        Protect against an invalid capacitance even though the runtime setter
        already rejects invalid values.
    */
    const double effective_C_J =
        (std::isfinite(c_j_per_k_) && c_j_per_k_ > 0.0)
            ? c_j_per_k_
            : 800.0;

    const double effective_h_WK =
        (std::isfinite(h_w_per_k_) && h_w_per_k_ > 0.0)
            ? h_w_per_k_
            : 0.8;

    /*
        Signed first order thermal exchange with the environment.
    */
    last_p_loss_W_ = effective_h_WK * (temperature_ - ambient_temp_K_);

    /*
        Net power into the source node.
    */
    last_net_W_ = pin_W + solar_absorbed_power_W_ - last_p_loss_W_;

    /*
        Temperature integration for the lumped thermal node.
    */
    const double dT_K = (last_net_W_ / effective_C_J) * dt_pos;
    temperature_ += dT_K;

    /*
        Repair invalid or unphysical states.
    */
    if (!std::isfinite(temperature_)) {
        temperature_ = ambient_temp_K_;
    }

    if (temperature_ < 0.0) {
        temperature_ = 0.0;
    }

    /*
        Store delivered source heater power for diagnostics and logging.
    */
    last_heat_W_  = pin_W;
    heat_input_w_ = pin_W;
}

void EffusionCell::setTargetTempK(double T_K) {
    /*
        Invalid targets fall back to idle baseline.

        A target near idle is treated as non meaningful by readiness logic.
    */
    if (!std::isfinite(T_K) || T_K < 0.0) {
        target_temp_K_ = kIdleBaselineTempK;
        return;
    }

    /*
        SourceInventory depletion bias: an aging crucible needs a hotter
        source for the same flux. Only meaningful process targets are
        biased; idle-level targets stay untouched so idle behavior is
        unchanged.
    */
    if (T_K > 400.0) {
        target_temp_K_ = T_K * (1.0 + inventory_temp_bias_frac_);
    } else {
        target_temp_K_ = T_K;
    }
}

bool EffusionCell::hasMeaningfulTarget() const {
    /*
        A source target is meaningful only when it is clearly above the idle
        baseline.
    */
    if (!std::isfinite(target_temp_K_)) {
        return false;
    }

    return target_temp_K_ > (kIdleBaselineTempK + kMeaningfulTargetMarginK);
}

bool EffusionCell::isAtTarget(double readiness_fraction) const {
    /*
        Preserve legacy one sided readiness behavior.

        If there is no meaningful target, the source should not block the
        scheduler.
    */
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
    /*
        Idle targets do not block scheduler progress.
    */
    if (!hasMeaningfulTarget()) {
        return ThermalBandState::Idle;
    }

    /*
        Invalid source state is treated conservatively as below target.
    */
    if (!std::isfinite(temperature_) || !std::isfinite(target_temp_K_)) {
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
    /*
        Ask the full band classifier for a below target result.
    */
    return getThermalBandState(
               lower_readiness_fraction,
               std::max(1.05, lower_readiness_fraction))
           == ThermalBandState::BelowTargetBand;
}

bool EffusionCell::isWithinTargetBand(double lower_readiness_fraction,
                                      double upper_readiness_fraction) const {
    /*
        Ask the full band classifier for an in band result.
    */
    return getThermalBandState(lower_readiness_fraction,
                               upper_readiness_fraction)
           == ThermalBandState::WithinTargetBand;
}

bool EffusionCell::isAboveTargetBand(double upper_readiness_fraction) const {
    /*
        The lower bound is fixed to the standard readiness fraction here
        because this helper only cares about the too hot side of the band.
    */
    double upper = upper_readiness_fraction;
    if (!std::isfinite(upper)) {
        upper = 1.05;
    }

    upper = std::max(upper, 0.90);

    return getThermalBandState(0.90, upper)
           == ThermalBandState::AboveTargetBand;
}