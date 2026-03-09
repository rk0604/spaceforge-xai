// Sim/src/EffusionCell.cpp
#include "EffusionCell.hpp"
#include "Logger.hpp"
#include "WakeChamber.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Avoid duplicate logs per process if tick() is called twice on one rank
static int s_last_logged_tick = -1;

// Gate streaks are computed in main.cpp; we mirror them here so they appear
// in EffusionCell.csv without changing any public API.
int g_underflux_streak_for_log = 0;
int g_temp_miss_streak_for_log = 0;

void EffusionCell::initialize() {
    last_heat_W_   = 0.0;
    heat_input_w_  = 0.0;
    last_p_loss_W_ = 0.0;
    last_net_W_    = 0.0;
    
    // Keep the target equal to the initial temperature at startup.
    target_temp_K_    = temperature_;
    last_pushed_temp_ = temperature_;
}

void EffusionCell::tick(const TickContext& ctx) {
    // De-dupe on this process
    if (ctx.tick_index == s_last_logged_tick) return;
    s_last_logged_tick = ctx.tick_index;

    // Log the original columns + the new physical diagnostic columns
    Logger::instance().log_wide(
        "EffusionCell",
        ctx.tick_index,
        ctx.time,
        { "status", "act_temp_K", "target_temp_K", "heatInput_w",
          "underflux_streak", "temp_miss_streak", 
          "P_loss_W", "P_net_W", "C_J", "h_WK" }, 
        { 1.0,
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

    // Optionally push updated setpoint into SPARTA at a cadence.
    if (sparta_ctrl_
        && (ctx.tick_index % push_every_ticks_ == 0)
        && std::fabs(temperature_ - last_pushed_temp_) >= push_threshold_K_) {
        sparta_ctrl_->setParameter("cell_temp_K", temperature_);
        sparta_ctrl_->markDirtyReload();
        last_pushed_temp_ = temperature_;
    }

    // Reset reported heat input for the next tick
    heat_input_w_ = 0.0;
}

void EffusionCell::shutdown() {
    // No special rows on shutdown
}

void EffusionCell::applyHeat(double watts, double dt) {
    const double T_env_K = 300.0; // ambient/environment temperature

    double Pin = std::max(0.0, watts);
    const double dt_pos = std::max(0.0, dt);

    // Calculate current heat loss and net power available for heating
    last_p_loss_W_ = h_w_per_k_ * (temperature_ - T_env_K);
    last_net_W_    = Pin - last_p_loss_W_;

    // Advance temperature state using the member variables (Synced to 800.0 / 0.8)
    const double dT = (last_net_W_ / c_j_per_k_) * dt_pos;
    temperature_ += dT;

    // Basic safety clamps
    if (!std::isfinite(temperature_)) temperature_ = T_env_K;
    if (temperature_ < 0.0)           temperature_ = 0.0;

    last_heat_W_  = Pin;
    heat_input_w_ = Pin; 
}