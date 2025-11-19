#ifndef ORBIT_HPP
#define ORBIT_HPP

// Simple circular orbit model for a single spacecraft.
// This header only defines data structures and the public API.
// See orbit.cpp for the function bodies.
//
// Assumptions:
// - Earth centered inertial (ECI) frame
// - Circular orbit with fixed semi major axis a
// - Optional inclination relative to equatorial plane
// - Simple Sun direction in the same frame for eclipse checks

#include <cstddef>

struct OrbitState {
    // Simulation time since t0 (seconds)
    double t_orbit_s = 0.0;

    // Angle along the orbit plane (radians).
    // For a circular orbit this is the true anomaly.
    double theta_rad = 0.0;

    // Position in ECI frame (meters)
    double x_m  = 0.0;
    double y_m  = 0.0;
    double z_m  = 0.0;

    // Velocity in ECI frame (meters per second)
    double vx_mps = 0.0;
    double vy_mps = 0.0;
    double vz_mps = 0.0;

    // Simple sunlight flags based on geometry
    // in_sun  = true  -> full sunlight
    // in_sun  = false -> eclipse (umbra)
    bool in_sun = true;

    // Scalar multiplier for solar power models.
    // 1.0 = full sun, 0.0 = full eclipse.
    // You can later extend this to partial eclipse.
    double solar_scale = 1.0;
};

class OrbitModel {
public:
    // Construct an orbit model for a circular orbit.
    //
    // altitude_m      - orbit altitude above mean Earth radius (meters)
    // dt_s            - time step per tick (seconds)
    // inclination_rad - orbit inclination relative to equator (radians).
    //                   0.0 means equatorial.
    // sun_theta_rad   - angle of the Sun direction in the reference plane
    //                   (radians). For a very simple model this can be fixed.
    OrbitModel(double altitude_m,
               double dt_s,
               double inclination_rad = 0.0,
               double sun_theta_rad   = 0.0);

    // Reset the orbit state to a chosen initial angle and time.
    //
    // t0_s        - initial orbit time (seconds since some epoch)
    // theta0_rad  - initial true anomaly angle (radians)
    void reset(double t0_s = 0.0, double theta0_rad = 0.0);

    // Advance the orbit by one time step dt_s.
    // This will:
    // - increment t_orbit_s
    // - update theta_rad
    // - recompute position and velocity in ECI
    // - recompute simple sunlight and solar_scale
    void step();

    // Recompute all derived quantities from the current theta_rad and t_orbit_s.
    // This can be used after changing parameters externally.
    void recompute_state();

    // Access the current orbit state (read only).
    const OrbitState &state() const { return state_; }

    // Get the current semi major axis (meters).
    double semi_major_axis_m() const { return a_m_; }

    // Get the orbital period (seconds).
    double period_s() const { return period_s_; }

    // Get the mean motion (radians per second).
    double mean_motion_rad_s() const { return n_rad_s_; }

    // Change the Sun direction angle used for simple eclipse checks.
    //
    // sun_theta_rad - angle in the reference plane, radians.
    void set_sun_theta(double sun_theta_rad);

    // Change the inclination used for mapping orbital plane -> ECI.
    //
    // inclination_rad - inclination in radians.
    void set_inclination(double inclination_rad);

    // Change the time step used in step().
    //
    // dt_s - new time step in seconds.
    void set_dt(double dt_s);

private:
    // Internal helper to recompute quantities that depend on a_m_.
    // Called from the constructor and any time the altitude / semi major axis
    // is changed in the future.
    void update_orbit_parameters();

    // Physical constants
    double Re_m_;      // Earth radius (meters)
    double mu_m3_s2_;  // Earth gravitational parameter mu (m^3 / s^2)

    // Orbit parameters
    double altitude_m_;     // altitude above Re_m_ (meters)
    double a_m_;            // semi major axis (meters)
    double n_rad_s_;        // mean motion (radians per second)
    double period_s_;       // orbital period (seconds)
    double inclination_rad_; // inclination (radians)

    // Time step used per call to step()
    double dt_s_;

    // Sun direction in the reference plane (radians)
    double sun_theta_rad_;

    // Current dynamic state
    OrbitState state_;
};

#endif // ORBIT_HPP
