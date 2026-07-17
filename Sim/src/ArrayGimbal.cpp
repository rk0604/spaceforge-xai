#include "ArrayGimbal.hpp"

#include "Logger.hpp"
#include "PowerBus.hpp"
#include "SolarArray.hpp"

#include <algorithm>
#include <cmath>

/*
    ArrayGimbal

    See ArrayGimbal.hpp for the model summary.

    Tick ordering contract

    Ticks BEFORE SolarArray so the pointing efficiency it pushes applies to
    this tick's array output.
*/

// Global sunlight scale driven by OrbitModel in main.cpp (same pattern as
// SolarArray.cpp). 0.0 = full eclipse, 1.0 = full sun.
extern double g_orbit_solar_scale;

namespace {
constexpr double kPi    = 3.141592653589793;
constexpr double kTwoPi = 2.0 * kPi;

// Wrap an angle into (-pi, +pi].
double wrapPi(double a) {
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0) a += kTwoPi;
    return a - kPi;
}

// Wrap an angle into [0, 2pi).
double wrapTwoPi(double a) {
    a = std::fmod(a, kTwoPi);
    if (a < 0.0) a += kTwoPi;
    return a;
}
} // namespace

void ArrayGimbal::initialize() {
    sun_angle_rad_    = 0.0;
    gimbal_angle_rad_ = 0.0;
    pointing_err_rad_ = 0.0;
    pointing_eff_     = 1.0;

    if (array_) {
        array_->setPointingEfficiency(pointing_eff_);
    }

    Logger::instance().log_wide(
        "ArrayGimbal", 0, 0.0,
        {
            "status",
            "sun_angle_deg",
            "gimbal_angle_deg",
            "pointing_err_deg",
            "pointing_eff",
            "slew_rate_deg_s",
            "in_sun",
            "power_req_W",
            "power_granted_W"
        },
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}
    );
}

void ArrayGimbal::tick(const TickContext& ctx) {
    const double solar_scale =
        std::isfinite(g_orbit_solar_scale)
            ? std::clamp(g_orbit_solar_scale, 0.0, 1.0)
            : 0.0;

    const bool in_sun = (solar_scale > eclipse_threshold_);

    // The Sun direction always advances in the body frame.
    const double omega_orbit = kTwoPi / orbit_period_s_;
    sun_angle_rad_ = wrapTwoPi(sun_angle_rad_ + omega_orbit * ctx.dt);

    // Track while sunlit; park during eclipse.
    double step_rad = 0.0;

    if (in_sun) {
        const double err = wrapPi(sun_angle_rad_ - gimbal_angle_rad_);
        const double max_step = omega_max_rad_s_ * ctx.dt;
        step_rad = std::clamp(err, -max_step, max_step);
        gimbal_angle_rad_ = wrapTwoPi(gimbal_angle_rad_ + step_rad);
    }

    pointing_err_rad_ = wrapPi(sun_angle_rad_ - gimbal_angle_rad_);

    // Cosine-loss pointing efficiency; the array cannot produce negative
    // power when facing away.
    pointing_eff_ = std::max(0.0, std::cos(pointing_err_rad_));

    if (array_) {
        array_->setPointingEfficiency(in_sun ? pointing_eff_ : 1.0);
    }

    // Electrical draw: slew effort while sunlit, park power in eclipse.
    const double max_step = omega_max_rad_s_ * ctx.dt;
    const double slew_frac =
        (max_step > 0.0) ? std::fabs(step_rad) / max_step : 0.0;

    const double power_req_W =
        in_sun ? (p_track_W_ + p_slew_W_ * slew_frac) : p_park_W_;

    double power_granted_W = 0.0;
    if (bus_) {
        power_granted_W = bus_->drawPower(power_req_W, ctx);
    }

    const double rad2deg = 180.0 / kPi;
    const double slew_rate_deg_s = (step_rad / ctx.dt) * rad2deg;

    Logger::instance().log_wide(
        "ArrayGimbal", ctx.tick_index, ctx.time,
        {
            "status",
            "sun_angle_deg",
            "gimbal_angle_deg",
            "pointing_err_deg",
            "pointing_eff",
            "slew_rate_deg_s",
            "in_sun",
            "power_req_W",
            "power_granted_W"
        },
        {
            1.0,
            sun_angle_rad_ * rad2deg,
            gimbal_angle_rad_ * rad2deg,
            pointing_err_rad_ * rad2deg,
            pointing_eff_,
            slew_rate_deg_s,
            in_sun ? 1.0 : 0.0,
            power_req_W,
            power_granted_W
        }
    );
}

void ArrayGimbal::shutdown() {}
