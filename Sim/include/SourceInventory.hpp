#pragma once

#include <algorithm>
#include <cmath>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class EffusionCell;

/*
    SourceInventory

    Depleting effusion-cell crucible charge.

    The crucible starts with a finite mass of source material. Live
    deposition consumes it in proportion to the delivered wafer flux, and a
    small idle sublimation loss applies whenever the source is held hot with
    the shutter closed. As the charge depletes, two real physical effects
    feed back into the EffusionCell thermal model:

    1. The lumped thermal capacitance falls (less melt mass), so the source
       heats and cools faster late in a run:

           C_eff = C_0 * (c_floor + (1 - c_floor) * frac)

    2. The source temperature required to sustain a given flux rises (falling
       melt level and depleted free surface), expressed as a fractional bias
       applied to every meaningful scheduler target:

           T_target_eff = T_target * (1 + beta * (1 - frac))

    Depletion model

        frac = m_remaining / m_initial            (clamped to [0, 1])

        live deposition (beam on, flux > 0):
            dm/dt = k_dep * (flux / 1e13)          [grams per minute]

        idle sublimation (T_cell > T_subl, beam off):
            dm/dt = r_subl                         [grams per minute]

    This node draws no electrical power itself; its impact arrives entirely
    through the extra heater energy the EffusionCell needs late in a run.
*/
class SourceInventory : public Subsystem {
public:
    SourceInventory() : Subsystem("SourceInventory") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setEffusionCell(EffusionCell* cell) { cell_ = cell; }

    /*
        Per-tick process state pushed from the scheduler loop in main.cpp,
        at the same site as GrowthMonitor::setBeamState.
    */
    void setProcessState(double flux_cm2s, bool beam_on) {
        process_flux_cm2s_ =
            (std::isfinite(flux_cm2s) && flux_cm2s > 0.0) ? flux_cm2s : 0.0;
        beam_on_ = beam_on;
    }

private:
    EffusionCell* cell_ = nullptr;

    // Crucible charge state (grams).
    double initial_g_{100.0};
    double remaining_g_{100.0};

    // Baseline EffusionCell constants captured at initialize() so the
    // capacitance scaling composes with runtime CLI configuration.
    double c0_J_per_K_{800.0};
    double h0_W_per_K_{0.8};

    // Scheduler-pushed process state.
    double process_flux_cm2s_{0.0};
    bool   beam_on_{false};

    // ---- Model constants ----

    // Depletion rate during live deposition, grams per minute at 1e13 flux.
    double k_dep_g_per_min_at_1e13_{0.15};

    // Idle sublimation loss while hot with the beam off (grams per minute),
    // and the source temperature above which it applies (K).
    double r_subl_g_per_min_{0.003};
    double t_subl_K_{900.0};

    // Thermal capacitance floor as the crucible empties.
    double c_floor_frac_{0.55};

    // Maximum fractional target-temperature bias at an empty crucible.
    double beta_bias_frac_{0.06};
};
