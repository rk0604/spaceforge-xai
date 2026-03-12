// Sim/include/SubstrateHeater.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Subsystem.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"

class SubstrateHeater : public Subsystem {
public:
  explicit SubstrateHeater(double maxPower_W = 2200.0, double wafer_radius_m = 0.15);

  // Only the leader should emit CSV rows to avoid duplicate logs.
  void setIsLeader(bool isLeader) { is_leader_ = isLeader; }

  // Update the current process-job state.
  //
  // Important:
  // - raw_job_flux_cm2s is the physical deposition request from the jobs file.
  // - This value must NOT be the SPARTA legality floor.
  // - Zero-flux jobs intentionally map to an idle substrate target for now.
  //
  // Current interim design:
  // - If job_active is false, target returns to idle baseline.
  // - If job_active is true and raw_job_flux_cm2s > 0, substrate target is
  //   derived from the requested deposition flux.
  // - If job_active is true and raw_job_flux_cm2s <= 0, substrate target stays
  //   at idle baseline because the jobs file does not yet expose an explicit
  //   substrate setpoint for prep or soak windows.
  void setJobState(int job_index, bool job_active, double raw_job_flux_cm2s);

  // Compute how much substrate-heater power is requested this tick.
  double computePowerRequestW();

  // Apply delivered heater power over dt seconds.
  void applyHeat(double watts, double dt_s);

  // Readiness and failure state used by main.cpp.
  bool isAtTarget() const;
  bool jobFailed() const { return job_failed_; }

  // Diagnostics accessors.
  double substrateTempK() const { return T_sub_K_; }
  double targetTempK() const { return T_target_K_; }
  double requestedPowerW() const { return P_requested_W_; }
  double deliveredPowerW() const { return P_delivered_W_; }
  int tempMissStreak() const { return temp_miss_streak_; }
  bool jobActive() const { return job_active_; }

  void initialize() override;
  void tick(const TickContext& ctx) override;
  void shutdown() override;

private:
  // Geometry and limits.
  double wafer_radius_m_;
  double wafer_area_m2_;
  double maxPower_W_;

  // Thermal state.
  double T_sub_K_    = 300.0;
  double T_target_K_ = 300.0;
  double T_env_K_    = 300.0;

  // Power bookkeeping.
  double P_requested_W_ = 0.0;
  double P_delivered_W_ = 0.0;
  double last_P_loss_W_ = 0.0;

  // Job state and health tracking.
  int  job_index_         = -1;
  bool job_active_        = false;
  int  temp_miss_streak_  = 0;
  bool job_failed_        = false;
  bool is_leader_         = false;

  // Physical constants.
  static constexpr double sigma_      = 5.670374419e-8;
  static constexpr double emissivity_ = 0.80;
  static constexpr double h_cond_WK_  = 0.50;
  static constexpr double C_J_per_K_  = 1500.0;
  static constexpr double Kp_W_per_K_ = 25.0;

  // Gate and failure thresholds.
  static constexpr double READY_BAND_K_     = 5.0;
  static constexpr int    FAIL_LIMIT_TICKS_ = 20;

  // Map physical deposition flux to a substrate growth target temperature.
  // This mapping is only meaningful for real deposition jobs.
  double fluxToTargetTemp(double raw_job_flux_cm2s) const;

  // Estimate total thermal losses at a given substrate temperature.
  double lossPowerW(double T_K) const;
};