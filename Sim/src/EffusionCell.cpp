#include "EffusionCell.hpp"
#include "Logger.hpp"
#include "WakeChamber.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Avoid duplicate logs per process if tick() is called twice on one rank
static int s_last_logged_tick = -1;

void EffusionCell::initialize() {
    last_heat_W_  = 0.0;
    heat_input_w_ = 0.0;
    // temperature_ is your internal crucible temp (K), default from header (e.g., 300 K)
}

void EffusionCell::tick(const TickContext& ctx) {
    // De-dupe on this process
    if (ctx.tick_index == s_last_logged_tick) return;
    s_last_logged_tick = ctx.tick_index;

    // One tidy, wide row per simulation tick
    Logger::instance().log_wide(
        "EffusionCell",
        ctx.tick_index,
        ctx.time,
        {"status","temperature","heatInput"},
        { 1.0,     temperature_,  heat_input_w_ }
    );

    // Optionally push updated setpoint into SPARTA at a cadence
    if (sparta_ctrl_
        && (ctx.tick_index % push_every_ticks_ == 0)
        && std::fabs(temperature_ - last_pushed_temp_) >= push_threshold_K_) {
        sparta_ctrl_->setParameter("cell_temp_K", temperature_);
        sparta_ctrl_->markDirtyReload();
        last_pushed_temp_ = temperature_;
    }

    // Display-only: reset reported heat input so it reflects this tick's command
    heat_input_w_ = 0.0;
}

void EffusionCell::shutdown() {
    // No special rows on shutdown
}

void EffusionCell::applyHeat(double watts, double dt) {
    // Physical-ish RC model: C dT/dt = P_in - h (T - T_env)
    // Tune these three constants to get the dynamics you want.
    const double C_J_per_K = 500.0;   // thermal capacitance of the cell
    const double h_W_per_K = 2.5;     // heat loss to environment
    const double T_env_K   = 300.0;   // ambient/environment temperature

    watts = std::max(0.0, watts);
    const double dt_pos = std::max(0.0, dt);

    const double net_W = watts - h_W_per_K * (temperature_ - T_env_K);
    const double dT    = (net_W / C_J_per_K) * dt_pos;

    temperature_ += dT;

    // Basic safety clamps
    if (!std::isfinite(temperature_)) temperature_ = T_env_K;
    if (temperature_ < 0.0)           temperature_ = 0.0;

    last_heat_W_  = watts;
    heat_input_w_ = watts; // logged on this tick; cleared at end of tick()
}
