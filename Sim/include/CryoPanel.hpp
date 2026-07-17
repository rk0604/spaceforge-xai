#pragma once

#include <algorithm>
#include <cmath>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class PowerBus;
class Radiator;

/*
    CryoPanel

    Cryopump cold panel driven by a mechanical cryocooler, with periodic
    regeneration bursts.

    The cryocooler is a genuine power hog that competes with both heaters on
    the shared bus. Its cold tip pumps condensable species (dominated by
    outgassed H2O and growth-related background) onto the panel; the adsorbed
    mass accumulates until a threshold forces a regeneration cycle, during
    which pumping stops and a regen heater bakes the panel while the
    cryocooler is off.

    Model summary (COOLING mode)

        duty      = clamp01((T_cold - T_duty_lo) / (T_duty_hi - T_duty_lo))

        P_req     = P_standby + (P_full - P_standby) * duty

        Q_lift    = Q_lift_max * (P_granted / P_full)

        Q_par     = h_par * (T_env - T_cold)
        T_env     = T_env_night + (T_env_day - T_env_night) * solar_scale

        dT_cold / dt = (Q_par - Q_lift) / C_cold

        Adsorption while T_cold < T_capture:
            dm/dt = r_base + r_flux * (flux / 1e13)   [grams per minute,
                                                       flux term only while
                                                       the beam is on]

    Regeneration (m_ads >= m_regen_threshold) lasts regen_duration_ticks:

        cryocooler off, regen heater draws P_regen
        dT_cold / dt = (Q_par + f_regen * P_regen_granted) / C_cold
        m_ads decays linearly to m_residual over the regen window
        no pumping during regen

    Heat rejected to the radiator loop every tick:

        COOLING: P_granted + Q_lift   (compressor electrical + lifted heat)
        REGEN:   P_regen_granted * (1 - f_regen)
*/
class CryoPanel : public Subsystem {
public:
    CryoPanel() : Subsystem("CryoPanel") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus)  { bus_      = bus; }
    void setRadiator(Radiator* rad)  { radiator_ = rad; }

    /*
        Per-tick process state pushed from the scheduler loop in main.cpp,
        at the same site as GrowthMonitor::setBeamState.

        flux_cm2s
            Current requested wafer flux.

        beam_on
            True while the MBE beam is live.
    */
    void setProcessState(double flux_cm2s, bool beam_on) {
        process_flux_cm2s_ =
            (std::isfinite(flux_cm2s) && flux_cm2s > 0.0) ? flux_cm2s : 0.0;
        beam_on_ = beam_on;
    }

private:
    PowerBus* bus_      = nullptr;
    Radiator* radiator_ = nullptr;

    // Cold tip / panel temperature in kelvin. Starts warm.
    double cold_temp_K_{300.0};

    // Accumulated condensed mass on the panel (grams).
    double adsorbed_g_{0.0};

    // Scheduler-pushed process state.
    double process_flux_cm2s_{0.0};
    bool   beam_on_{false};

    // Regeneration state. 0 means COOLING mode.
    int regen_ticks_left_{0};

    // ---- Model constants ----

    // Cold head lumped capacitance (J per K).
    double c_cold_J_per_K_{2000.0};

    // Parasitic coupling to the surrounding environment (W per K).
    double h_par_W_per_K_{0.3};

    // Effective environment temperature seen by the cold head (K).
    double env_night_K_{250.0};
    double env_day_K_{325.0};

    // Cryocooler electrical envelope (W) and thermal lift at full power (W).
    double p_standby_W_{30.0};
    double p_full_W_{150.0};
    double q_lift_max_W_{60.0};

    // Duty throttle band: full power above hi, standby below lo (K).
    double t_duty_lo_K_{110.0};
    double t_duty_hi_K_{130.0};

    // Pumping only happens below this capture temperature (K).
    double t_capture_K_{140.0};

    // Adsorption rates in grams per minute.
    double r_base_g_per_min_{0.010};
    double r_flux_g_per_min_at_1e13_{0.012};

    // Regeneration trigger, duration, heater power, and residual mass.
    double m_regen_threshold_g_{12.0};
    int    regen_duration_ticks_{25};
    double p_regen_W_{200.0};
    double m_residual_g_{0.5};

    // Linear desorption rate that empties threshold -> residual across the
    // regen window. Derived once here so the decay invariant lives next to
    // the constants it depends on (declaration order matters).
    double regen_release_g_per_tick_{
        (m_regen_threshold_g_ - m_residual_g_) /
        static_cast<double>(regen_duration_ticks_)};

    // Fraction of regen heater power that heats the cold head; the rest is
    // rejected to the radiator loop.
    double f_regen_{0.5};
};
