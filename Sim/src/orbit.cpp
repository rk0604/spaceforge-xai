#include "orbit.hpp"

#include <cmath>

// Simple local constant for pi so we do not rely on M_PI.
static constexpr double ORBIT_PI       = 3.141592653589793;
// Forced orbital period: 94 minutes (in seconds).
static constexpr double ORBIT_PERIOD_S = 94.0 * 60.0;

// Constructor
OrbitModel::OrbitModel(double altitude_m,
                       double dt_s,
                       double inclination_rad,
                       double sun_theta_rad)
    : Re_m_(6371e3),              // mean Earth radius in meters
      mu_m3_s2_(3.986004418e14),  // Earth mu in m^3 / s^2
      altitude_m_(altitude_m),
      a_m_(0.0),
      n_rad_s_(0.0),
      period_s_(0.0),
      inclination_rad_(inclination_rad),
      dt_s_(dt_s),
      sun_theta_rad_(sun_theta_rad),
      state_()
{
    // Semi-major axis = Earth radius + altitude (kept for geometry)
    a_m_ = Re_m_ + altitude_m_;

    // Compute n_rad_s_ and period_s_ (will be forced to 94 min).
    update_orbit_parameters();

    // Start at t = 0, theta = 0 by default
    state_.t_orbit_s = 0.0;
    state_.theta_rad = 0.0;

    // Make sure all derived quantities are valid
    recompute_state();
}

// Reset the orbit state to a specific initial time and angle.
void OrbitModel::reset(double t0_s, double theta0_rad)
{
    state_.t_orbit_s = t0_s;

    // Normalize theta into [0, 2*pi)
    double two_pi = 2.0 * ORBIT_PI;
    state_.theta_rad = std::fmod(theta0_rad, two_pi);
    if (state_.theta_rad < 0.0) {
        state_.theta_rad += two_pi;
    }

    recompute_state();
}

// Advance the orbit by one dt_s step.
void OrbitModel::step()
{
    state_.t_orbit_s += dt_s_;

    double two_pi = 2.0 * ORBIT_PI;
    state_.theta_rad += n_rad_s_ * dt_s_;
    state_.theta_rad = std::fmod(state_.theta_rad, two_pi);
    if (state_.theta_rad < 0.0) {
        state_.theta_rad += two_pi;
    }

    recompute_state();
}

// Recompute position, velocity, and sunlight based on theta_rad and t_orbit_s.
void OrbitModel::recompute_state()
{
    // 1) Position in orbital plane (circular orbit: r = a_m_)
    double r  = a_m_;
    double ct = std::cos(state_.theta_rad);
    double st = std::sin(state_.theta_rad);

    double x_orb = r * ct;
    double y_orb = r * st;
    double z_orb = 0.0;

    // 2) Rotate by inclination about the x-axis to get ECI coordinates
    double ci = std::cos(inclination_rad_);
    double si = std::sin(inclination_rad_);

    double x_eci = x_orb;
    double y_eci = y_orb * ci;
    double z_eci = y_orb * si;

    state_.x_m = x_eci;
    state_.y_m = y_eci;
    state_.z_m = z_eci;

    // 3) Velocity in orbital plane
    // v magnitude = r * n for circular orbit
    double vx_orb = -r * n_rad_s_ * st;
    double vy_orb =  r * n_rad_s_ * ct;
    double vz_orb = 0.0;

    double vx_eci = vx_orb;
    double vy_eci = vy_orb * ci;
    double vz_eci = vy_orb * si;

    state_.vx_mps = vx_eci;
    state_.vy_mps = vy_eci;
    state_.vz_mps = vz_eci;

    // 4) Simple sunlight / eclipse geometry.

    // Sun is a unit vector in the reference plane at angle sun_theta_rad_.
    double cs = std::cos(sun_theta_rad_);
    double ss = std::sin(sun_theta_rad_);

    double sun_x = cs;
    double sun_y = ss;
    double sun_z = 0.0;

    // Dot product between position vector and sun direction.
    double dot_rs = x_eci * sun_x + y_eci * sun_y + z_eci * sun_z;

    // Cosine of angle between r and sun.
    double rmag = std::sqrt(x_eci * x_eci + y_eci * y_eci + z_eci * z_eci);
    double cos_alpha = 0.0;
    if (rmag > 0.0) {
        cos_alpha = dot_rs / rmag;
    }

    // "In sun" if the sub-solar point is on the same side of Earth as the spacecraft.
    bool in_sun = (cos_alpha > 0.0);
    state_.in_sun = in_sun;

    // 5) Time-based sinusoidal solar scale, locked to the 94-min period.
    //
    // We define an orbit phase phi in [0, 2*pi) from the elapsed orbit time:
    //   phi = 2*pi * ( (t mod period) / period )
    //
    // Then a simple cosine:
    //   s_phase = 0.5 * (1 + cos(phi))
    //
    // has:
    //   - s_phase = 1 at t = 0 (start of orbit)
    //   - s_phase = 0 at half-orbit
    //   - 94-minute period, same as SPARTA CUP_SCALE if you mirror this formula.
    //
    // Finally we gate it by the geometry-based eclipse:
    //   - if in_sun is false, we force s = 0
    double s_phase = 0.0;
    if (period_s_ > 0.0) {
        double t_mod = std::fmod(state_.t_orbit_s, period_s_);
        if (t_mod < 0.0) {
            t_mod += period_s_;
        }
        double phi = 2.0 * ORBIT_PI * (t_mod / period_s_);
        s_phase = 0.5 * (1.0 + std::cos(phi));  // in [0, 1], starts at 1
    }

    double s = (in_sun ? s_phase : 0.0);

    // Clamp numerically just in case.
    if (s < 0.0) s = 0.0;
    if (s > 1.0) s = 1.0;

    state_.solar_scale = s;

    // NOTE:
    //  - state_.in_sun tells you the geometry (eclipse or not).
    //  - state_.solar_scale is a smooth sinusoid locked to the 94-min period,
    //    starting at 1 for t=0, and zeroed in eclipse. SPARTA's CUP_SCALE can
    //    replicate this exactly from (t, period_s_) without needing the full geometry.
}

// Change the Sun direction angle used for eclipse checks.
void OrbitModel::set_sun_theta(double sun_theta_rad)
{
    sun_theta_rad_ = sun_theta_rad;
    recompute_state();
}

// Change inclination used to map orbital plane into ECI.
void OrbitModel::set_inclination(double inclination_rad)
{
    inclination_rad_ = inclination_rad;
    recompute_state();
}

// Change the time step used by step().
void OrbitModel::set_dt(double dt_s)
{
    dt_s_ = dt_s;
}

// Internal helper to recompute n_rad_s_ and period_s_ when a_m_ changes.
void OrbitModel::update_orbit_parameters()
{
    // If you want the physically implied period from mu and a, this is it:
    // double n_kepler = std::sqrt(mu_m3_s2_ / (a_m_ * a_m_ * a_m_));
    // double T_kepler = 2.0 * ORBIT_PI / n_kepler;

    // For SpaceForgeOS we *force* a 94-minute orbit, independent of altitude.
    period_s_ = ORBIT_PERIOD_S;
    n_rad_s_  = 2.0 * ORBIT_PI / period_s_;
}
