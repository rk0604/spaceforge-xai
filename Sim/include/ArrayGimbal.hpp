#pragma once

#include <cmath>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class PowerBus;
class SolarArray;

/*
    ArrayGimbal

    Single-axis solar array drive that keeps the panel pointed at the Sun.

    In LEO the Sun direction rotates a full revolution per orbit in the
    spacecraft body frame, so the gimbal must track continuously while
    sunlit. During eclipse the drive parks (no tracking, minimal draw) while
    the geometric Sun direction keeps moving, so every dawn begins with a
    large pointing error and a re-acquisition slew that visibly depresses
    array output for several ticks.

    Model summary

        sun_angle    += omega_orbit * dt          (always advances, wraps 2pi)

        sunlit:
            err       = wrap(sun_angle - gimbal_angle)
            step      = clamp(err, -omega_max * dt, +omega_max * dt)
            gimbal   += step
        eclipse:
            gimbal parked (step = 0)

        pointing_err  = wrap(sun_angle - gimbal_angle)
        pointing_eff  = max(0, cos(pointing_err))   (pushed into SolarArray)

        P_req = sunlit:  P_track + P_slew * |step| / (omega_max * dt)
                eclipse: P_park

    Tick ordering and lag

    This node ticks AFTER SolarArray, so its electrical draw lands on a bus
    that already holds this tick's solar generation (instead of forcing a
    needless battery micro-discharge every tick). The pointing efficiency it
    pushes therefore applies to the NEXT tick's array output - a deliberate
    one-tick lag, consistent with the other feedback edges in the sim
    (battery derating, inventory target bias).

    The orbit period defaults to the 400 km value but should be synced from
    the real OrbitModel via setOrbitPeriodS so there is one source of truth.
*/
class ArrayGimbal : public Subsystem {
public:
    ArrayGimbal() : Subsystem("ArrayGimbal") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus)     { bus_   = bus; }
    void setSolarArray(SolarArray* arr) { array_ = arr; }

    /*
        Sync the Sun-sweep rate with the actual orbit model. main.cpp calls
        this with OrbitModel::period_s() so the gimbal's internal sun angle
        cannot drift against the real eclipse phase.
    */
    void setOrbitPeriodS(double period_s) {
        if (std::isfinite(period_s) && period_s > 60.0) {
            orbit_period_s_ = period_s;
        }
    }

private:
    PowerBus*   bus_   = nullptr;
    SolarArray* array_ = nullptr;

    // Geometric Sun direction and gimbal shaft angle (radians, wrapped).
    // These persist across ticks; everything else in tick() is derived.
    double sun_angle_rad_{0.0};
    double gimbal_angle_rad_{0.0};

    // ---- Model constants ----

    // Orbit period for the Sun direction sweep (seconds). Default matches
    // the 400 km OrbitModel; overridden at startup via setOrbitPeriodS.
    double orbit_period_s_{5560.0};

    // Maximum gimbal slew rate (radians per second). 0.3 deg/s.
    double omega_max_rad_s_{0.3 * 3.141592653589793 / 180.0};

    // Electrical draw envelope (watts).
    double p_track_W_{8.0};
    double p_slew_W_{45.0};
    double p_park_W_{2.0};

    // Solar scale below this is treated as eclipse for the drive logic.
    double eclipse_threshold_{0.05};
};
