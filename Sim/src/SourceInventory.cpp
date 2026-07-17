#include "SourceInventory.hpp"

#include "EffusionCell.hpp"
#include "Logger.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/*
    SourceInventory

    See SourceInventory.hpp for the model summary.

    Tick ordering contract

    Ticks after the EffusionCell so it reads this tick's settled source
    temperature. The capacitance scaling and target bias it pushes take
    effect from the next tick onward, which is the correct causality for a
    slowly depleting reservoir.
*/

namespace {

// One shared column list keeps the header row (initialize) and the data rows
// (tick) from ever drifting apart.
const std::vector<std::string> kColumns = {
    "status",
    "remaining_g",
    "remaining_frac",
    "deplete_rate_g_min",
    "live_deposition",
    "cell_temp_K",
    "c_eff_J_per_K",
    "temp_bias_frac"
};

} // namespace

void SourceInventory::initialize() {
    remaining_g_ = initial_g_;

    /*
        Capture the runtime-configured EffusionCell constants as the
        baseline. main.cpp applies CLI configuration before
        engine.initialize(), so these getters already reflect the tracker
        values for this config.
    */
    if (cell_) {
        c0_J_per_K_ = cell_->getThermalCapacitanceJPerK();
        h0_W_per_K_ = cell_->getHeatLossWPerK();
    }

    Logger::instance().log_wide(
        "SourceInventory", 0, 0.0, kColumns,
        {1.0, remaining_g_, 1.0, 0.0, 0.0, 300.0, c0_J_per_K_, 0.0}
    );
}

void SourceInventory::tick(const TickContext& ctx) {
    const double dt_min = ctx.dt / 60.0;

    const double cell_temp_K = cell_ ? cell_->getTemperatureK() : 300.0;

    // ---- 1) Depletion ----
    const bool live_deposition = beam_on_ && (process_flux_cm2s_ > 0.0);

    double deplete_rate_g_min = 0.0;

    if (live_deposition) {
        deplete_rate_g_min =
            k_dep_g_per_min_at_1e13_ * (process_flux_cm2s_ / 1.0e13);
    } else if (cell_temp_K > t_subl_K_) {
        // Hot source with a closed shutter still slowly loses material.
        deplete_rate_g_min = r_subl_g_per_min_;
    }

    remaining_g_ = std::max(0.0, remaining_g_ - deplete_rate_g_min * dt_min);

    const double frac = SimHelpers::clamp01(
        (initial_g_ > 0.0) ? (remaining_g_ / initial_g_) : 0.0);

    // ---- 2) Feed the depletion back into the source thermal model ----
    const double c_eff_J_per_K =
        c0_J_per_K_ * (c_floor_frac_ + (1.0 - c_floor_frac_) * frac);

    const double temp_bias_frac = beta_bias_frac_ * (1.0 - frac);

    if (cell_) {
        cell_->setThermalConstants(c_eff_J_per_K, h0_W_per_K_);
        cell_->setInventoryTempBiasFrac(temp_bias_frac);
    }

    Logger::instance().log_wide(
        "SourceInventory", ctx.tick_index, ctx.time, kColumns,
        {
            1.0,
            remaining_g_,
            frac,
            deplete_rate_g_min,
            live_deposition ? 1.0 : 0.0,
            cell_temp_K,
            c_eff_J_per_K,
            temp_bias_frac
        }
    );
}

void SourceInventory::shutdown() {}
