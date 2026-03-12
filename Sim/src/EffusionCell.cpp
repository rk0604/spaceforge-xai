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
constexpr double kAmbientTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;
}

void EffusionCell::initialize() {
    last_heat_W_       = 0.0;
    heat_input_w_      = 0.0;
    last_p_loss_W_     = 0.0;
    last_net_W_        = 0.0;

    // Keep the target equal to the initial temperature at startup so the cell
    // begins in an idle, already-satisfied state.
    target_temp_K_     = temperature_;
    last_pushed_temp_  = temperature_;
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
            heat_input_w_,
            static_cast<double>(g_underflux_streak_for_log),
            static_cast<double>(g_temp_miss_streak_for_log),
            last_p_loss_W_,
            last_net_W_,
            c_j_per_k_,
            h_w_per_k_
        }
    );

    // Optionally push the current actual cell temperature into SPARTA at a
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

void EffusionCell::applyHeat(double watts, double dt) {
    const double Pin    = std::max(0.0, watts);
    const double dt_pos = std::max(0.0, dt);

    // Heat loss grows linearly with temperature above ambient.
    last_p_loss_W_ = h_w_per_k_ * (temperature_ - kAmbientTempK);
    last_net_W_    = Pin - last_p_loss_W_;

    // First-order thermal update.
    const double dT = (last_net_W_ / c_j_per_k_) * dt_pos;
    temperature_ += dT;

    // Basic safety clamps.
    if (!std::isfinite(temperature_)) {
        temperature_ = kAmbientTempK;
    }
    if (temperature_ < 0.0) {
        temperature_ = 0.0;
    }

    last_heat_W_  = Pin;
    heat_input_w_ = Pin;
}

void EffusionCell::setTargetTempK(double T_K) {
    // Clamp invalid or non-physical targets back to the idle baseline.
    if (!std::isfinite(T_K) || T_K < 0.0) {
        target_temp_K_ = kAmbientTempK;
        return;
    }

    target_temp_K_ = T_K;
}

bool EffusionCell::hasMeaningfulTarget() const {
    if (!std::isfinite(target_temp_K_)) {
        return false;
    }

    return target_temp_K_ > (kAmbientTempK + kMeaningfulTargetMarginK);
}

bool EffusionCell::isAtTarget(double readiness_fraction) const {
    // Idle or non-meaningful targets should not block process readiness.
    if (!hasMeaningfulTarget()) {
        return true;
    }

    if (!std::isfinite(temperature_) || !std::isfinite(target_temp_K_)) {
        return false;
    }

    // Clamp the caller-provided fraction to a sensible range.
    double frac = readiness_fraction;
    if (!std::isfinite(frac)) {
        frac = 0.90;
    }
    frac = std::clamp(frac, 0.0, 1.0);

    return temperature_ >= (frac * target_temp_K_);
}