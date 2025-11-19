/*
| tick | t_orbit_min | theta_deg | in_sun | solar_scale |
| ---- | ----------- | --------- | ------ | ----------- |
| 0    | 0           | 0.0       | 1      | 1.0         |
| 1    | 1           | 4.0       | 1      | 1.0         |
| 2    | 2           | 8.0       | 1      | 1.0         |
| 3    | 3           | 12.0      | 1      | 1.0         |
| 4    | 4           | 16.0      | 1      | 1.0         |
| ...  | ...         | ...       | ...    | ...         |
|  22  |  22         | 88        | 1      | 1.0         |
|  23  |  23         | 92        | 0      | 0.0         |
| ...  | ...         | ...       | 0      | 0.0         |
|  67  |  67         | 268       | 0      | 0.0         |
|  68  |  68         |  272      | 1      | 1.0         |
*/

#include "orbit.hpp"

#include <cmath>

// Constructor
OrbitModel::OrbitModel(double altitude_m,
                       double dt_s,
                       double inclination_rad,
                       double sun_theta_rad)
    : Re_m_(6371e3),          // mean Earth radius in meters
      mu_m3_s2_(3.986004418e14), // Earth mu in m^3 / s^2
      altitude_m_(altitude_m),
      a_m_(0.0),
      n_rad_s_(0.0),
      period_s_(0.0),
      inclination_rad_(inclination_rad),
      dt_s_(dt_s),
      sun_theta_rad_(sun_theta_rad),
      state_()
{
    // TODO:
    // 1. Compute a_m_ = Re_m_ + altitude_m_.
    // 2. Call update_orbit_parameters() to compute n_rad_s_ and period_s_.
    // 3. Initialize state_ (t_orbit_s = 0, theta_rad = 0, etc).
    // 4. Call recompute_state() so that x,y,z and sunlight flags are valid.
}

// Reset the orbit state to a specific initial time and angle.
void OrbitModel::reset(double t0_s, double theta0_rad)
{
    // Inputs:
    // - t0_s: new starting orbit time in seconds.
    // - theta0_rad: starting angle along orbit in radians.
    //
    // Expected behavior:
    // - Set state_.t_orbit_s = t0_s.
    // - Set state_.theta_rad = normalized version of theta0_rad.
    // - Call recompute_state() to update x,y,z, velocity, and sunlight flags.
}

// Advance the orbit by one dt_s step.
void OrbitModel::step()
{
    // Inputs: none (uses internal dt_s_).
    //
    // Expected behavior:
    // - Increment state_.t_orbit_s by dt_s_.
    // - Increment state_.theta_rad by n_rad_s_ * dt_s_.
    // - Wrap theta into [0, 2*pi) using std::fmod.
    // - Call recompute_state() to update x,y,z, velocity, and sunlight.
}

// Recompute position, velocity, and sunlight based on theta_rad and t_orbit_s.
void OrbitModel::recompute_state()
{
    // Inputs:
    // - Uses current a_m_, inclination_rad_, sun_theta_rad_,
    //   and the current state_.theta_rad and state_.t_orbit_s.
    //
    // Expected behavior:
    // 1. Compute position in the orbital plane:
    //      r = a_m_
    //      x_orb = r * cos(theta_rad)
    //      y_orb = r * sin(theta_rad)
    //      z_orb = 0
    //
    // 2. Rotate this position by inclination_rad_ to get ECI coordinates.
    //    For an equatorial orbit (inclination_rad_ == 0), this is just:
    //      x = x_orb
    //      y = y_orb
    //      z = 0
    //
    // 3. Compute velocity in the orbital plane:
    //      vx_orb = -r * n_rad_s_ * sin(theta_rad)
    //      vy_orb =  r * n_rad_s_ * cos(theta_rad)
    //      vz_orb = 0
    //    Then rotate to ECI in the same way as position.
    //
    // 4. Store results in state_.x_m, y_m, z_m, vx_mps, vy_mps, vz_mps.
    //
    // 5. Simple sunlight model:
    //    - Treat the Sun as lying in the reference plane at angle sun_theta_rad_.
    //    - Compute the angle between spacecraft position vector and Sun vector.
    //    - If cos(angle) > 0, set in_sun = true, solar_scale = 1.0.
    //      Otherwise set in_sun = false, solar_scale = 0.0.
}

// Getters are already implemented inline in the header.

// Change the Sun direction angle used for eclipse checks.
void OrbitModel::set_sun_theta(double sun_theta_rad)
{
    // Inputs:
    // - sun_theta_rad: new Sun direction angle in radians.
    //
    // Expected behavior:
    // - Update sun_theta_rad_.
    // - Optionally call recompute_state() so that in_sun and solar_scale
    //   reflect the new Sun direction immediately.
}

// Change inclination used to map orbital plane into ECI.
void OrbitModel::set_inclination(double inclination_rad)
{
    // Inputs:
    // - inclination_rad: new orbit inclination in radians.
    //
    // Expected behavior:
    // - Update inclination_rad_.
    // - Call recompute_state() so that position and velocity are updated
    //   under the new inclination.
}

// Change the time step used by step().
void OrbitModel::set_dt(double dt_s)
{
    // Inputs:
    // - dt_s: new time step in seconds.
    //
    // Expected behavior:
    // - Update dt_s_.
    // - No change to state_ needed until the next call to step().
    dt_s_ = dt_s;
}

// Internal helper to recompute n_rad_s_ and period_s_ when a_m_ changes.
void OrbitModel::update_orbit_parameters()
{
    // Inputs:
    // - Uses current a_m_ and mu_m3_s2_.
    //
    // Expected behavior:
    // - Compute mean motion:
    //     n_rad_s_ = sqrt(mu_m3_s2_ / (a_m_ * a_m_ * a_m_));
    // - Compute orbital period:
    //     period_s_ = 2.0 * pi / n_rad_s_;
    //
    // Note:
    // - You will need M_PI or your own constant for pi.
}


