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

  void setIsLeader(bool isLeader) { is_leader_ = isLeader; }

  void setJobState(int job_index, bool job_active, double Fwafer_cm2s);
  double computePowerRequestW();
  void applyHeat(double watts, double dt_s);

  bool isAtTarget() const;
  bool jobFailed() const { return job_failed_; }

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
  double wafer_radius_m_;
  double wafer_area_m2_;
  double maxPower_W_;

  double T_sub_K_    = 300.0;
  double T_target_K_ = 300.0;
  double T_env_K_    = 300.0;

  double P_requested_W_ = 0.0;
  double P_delivered_W_ = 0.0;
  double last_P_loss_W_ = 0.0; // Added for logging consistency

  int  job_index_  = -1;
  bool job_active_ = false;
  int  temp_miss_streak_ = 0;
  bool job_failed_       = false;
  bool is_leader_        = false;

  static constexpr double sigma_      = 5.670374419e-8;
  static constexpr double emissivity_ = 0.80;
  static constexpr double h_cond_WK_  = 0.50;
  static constexpr double C_J_per_K_  = 1500.0;
  static constexpr double Kp_W_per_K_ = 25.0;

  static constexpr double READY_BAND_K_     = 5.0;
  static constexpr int    FAIL_LIMIT_TICKS_ = 20;

  double fluxToTargetTemp(double Fwafer_cm2s) const;
  double lossPowerW(double T_K) const;
};