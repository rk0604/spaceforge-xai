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
    // temperature_ is your internal crucible temp (K), default from header (e.g., 300 K)
    // Keep the target equal to the initial temperature at startup.
    target_temp_K_    = temperature_;
    last_pushed_temp_ = temperature_;
}

void EffusionCell::tick(const TickContext& ctx) {
    // De-dupe on this process
    if (ctx.tick_index == s_last_logged_tick) return;
    s_last_logged_tick = ctx.tick_index;

    // One tidy, wide row per simulation tick.
    // act_temp_K        = achieved crucible temperature this tick
    // target_temp_K     = desired crucible temperature implied by job flux
    // heatInput_w       = thermal power actually delivered this tick
    // underflux_streak  = consecutive ticks where bus under-supplied heater
    // temp_miss_streak  = consecutive ticks where temp was below target band
    Logger::instance().log_wide(
        "EffusionCell",
        ctx.tick_index,
        ctx.time,
        {"status", "act_temp_K", "target_temp_K", "heatInput_w",
         "underflux_streak", "temp_miss_streak"},
        { 1.0,
          temperature_,
          target_temp_K_,
          heat_input_w_,
          static_cast<double>(g_underflux_streak_for_log),
          static_cast<double>(g_temp_miss_streak_for_log) }
    );

    // Optionally push updated setpoint into SPARTA at a cadence.
    // For now we keep using the achieved temperature as the setpoint;
    // target_temp_K_ is purely for logging / analysis.
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
    // Tuned for kW-class heater so we can reach ~1100–1300 K in typical job windows.
    const double C_J_per_K = 1000.0;   // thermal capacitance of the cell
    const double h_W_per_K = 1.5;      // heat loss to environment
    const double T_env_K   = 300.0;    // ambient/environment temperature

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

    // NOTE: target_temp_K_ is NOT updated here.
    // It is driven externally (from main.cpp) based on the flux schedule.
}
