// Sim/include/SubstrateHeater.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Subsystem.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"

/*
SubstrateHeater
---------------
Purpose:
  Models wafer + chuck (substrate stack) temperature dynamics and readiness gating.

Power model:
  SubstrateHeater does not draw from PowerBus directly.
  HeaterBank allocates electrical power to the substrate heater and calls:
      applyHeat(P_alloc_W, dt_s)

State equation (lumped-parameter thermal model):
  C * dT/dt = P_in - P_loss(T)

  where:
    - C [J/K] is effective heat capacity of the substrate stack
    - P_in [W] is allocated heater power (>= 0)
    - P_loss(T) [W] models heat rejection to environment:
        P_loss(T) = P_rad(T) + P_cond(T)

Radiation loss:
  P_rad(T) = eps * sigma * A * (T^4 - T_env^4), clamped to >= 0

Conduction / effective linear loss:
  P_cond(T) = h * (T - T_env), clamped to >= 0

Control law (per-tick power request used by main.cpp):
  P_req = clamp( P_loss(T_current) + Kp * max(T_target - T_current, 0), 0, P_max )

Target temperature:
  Derived from commanded wafer flux Fwafer_cm2s via log-interpolation between anchor points.

Gating:
  isAtTarget() is true when:
    T_sub >= T_target - READY_BAND_K_
  while job_active is true.

Failure:
  temp_miss_streak increments each tick that the substrate remains below the READY_BAND.
  job_failed becomes true once temp_miss_streak reaches FAIL_LIMIT_TICKS_.

Logging:
  Writes wide rows using Logger under subsystem name "substrate" => substrate.csv.
  Logging is leader-only to avoid MPI file contention; setIsLeader(true) on rank 0.
*/

class SubstrateHeater : public Subsystem {
public:
  // 300 mm wafer => radius ~0.15 m
  explicit SubstrateHeater(double maxPower_W = 2200.0, double wafer_radius_m = 0.15);

  // Prevent multi-rank CSV corruption/duplication
  void setIsLeader(bool isLeader) { is_leader_ = isLeader; }

  // Updates job state and target temperature based on flux requirement.
  // job_index: index of active job (or -1 when idle)
  // job_active: whether job is currently active
  // Fwafer_cm2s: commanded wafer flux (cm^-2 s^-1), used to derive T_target
  void setJobState(int job_index, bool job_active, double Fwafer_cm2s);

  // Computes requested heater power for the current tick (W).
  // Stores the value internally for logging and for main.cpp to pass into HeaterBank.
  double computePowerRequestW();

  // Applies allocated heater power for this tick and advances temperature state.
  // watts: allocated heater power from HeaterBank (W)
  // dt_s: tick time step (s)
  void applyHeat(double watts, double dt_s);

  // Readiness gate
  bool isAtTarget() const;

  // Failure state (true once temperature misses exceed FAIL_LIMIT_TICKS_)
  bool jobFailed() const { return job_failed_; }

  // Diagnostics for main.cpp / logs
  double substrateTempK() const { return T_sub_K_; }
  double targetTempK() const { return T_target_K_; }
  double requestedPowerW() const { return P_requested_W_; }
  double deliveredPowerW() const { return P_delivered_W_; }
  int tempMissStreak() const { return temp_miss_streak_; }
  bool jobActive() const { return job_active_; }

  // Subsystem API
  void initialize() override;
  void tick(const TickContext& ctx) override;
  void shutdown() override;

private:
  // Geometry
  double wafer_radius_m_;
  double wafer_area_m2_;

  // Electrical limit for substrate heater allocation
  double maxPower_W_;

  // Thermal state (Kelvin)
  double T_sub_K_    = 300.0;  // current substrate temperature
  double T_target_K_ = 300.0;  // derived target temperature
  double T_env_K_    = 300.0;  // environment / sink temperature

  // Power bookkeeping (Watts) for logging
  double P_requested_W_ = 0.0; // computed request from control law
  double P_delivered_W_ = 0.0; // allocated power received from HeaterBank

  // Job state
  int  job_index_  = -1;
  bool job_active_ = false;

  // Failure / gating
  int  temp_miss_streak_ = 0;
  bool job_failed_       = false;

  // Logging control
  bool is_leader_ = false;

  // --- Physical/behavior constants ---
  // Stefan–Boltzmann constant [W m^-2 K^-4]
  static constexpr double sigma_      = 5.670374419e-8;
  // Effective emissivity (dimensionless)
  static constexpr double emissivity_ = 0.80;
  // Effective linear loss coefficient [W/K]
  static constexpr double h_cond_WK_  = 0.50;
  // Effective heat capacity [J/K]
  static constexpr double C_J_per_K_  = 1500.0;
  // Proportional control gain [W/K]
  static constexpr double Kp_W_per_K_ = 25.0;

  // Readiness band around target
  static constexpr double READY_BAND_K_     = 5.0;
  // Consecutive misses required to fail a job
  static constexpr int    FAIL_LIMIT_TICKS_ = 20;

  // Flux -> target temperature mapping (K)
  double fluxToTargetTemp(double Fwafer_cm2s) const;

  // Heat loss power model at temperature T_K (W), clamped to >= 0
  double lossPowerW(double T_K) const;
};
