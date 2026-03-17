#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Subsystem.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"

class SubstrateHeater : public Subsystem {
public:
  /*
      Scheduler-facing thermal band state.

      Why this exists:
      The scheduler needs to distinguish these cases cleanly:

      - BelowTargetBand
          More heating is still needed before the phase is thermally ready.

      - WithinTargetBand
          The substrate is thermally ready for the current recipe phase.

      - AboveTargetBand
          The substrate is too hot for the current recipe phase and must cool.

      - Idle
          No meaningful substrate-heating target is currently active.
  */
  enum class ThermalBandState {
    Idle = 0,
    BelowTargetBand,
    WithinTargetBand,
    AboveTargetBand
  };

  explicit SubstrateHeater(double maxPower_W = 2200.0, double wafer_radius_m = 0.15);

  // Only the leader should emit CSV rows to avoid duplicate logs.
  void setIsLeader(bool isLeader) { is_leader_ = isLeader; }

  /*
      Update the current scheduler-facing substrate control state.

      Inputs:
      - job_index
          Current controlling job identity.

      - job_active
          True when a controlling row owns substrate control this tick.

      - raw_job_flux_cm2s
          Physical deposition flux request from the schedule row.
          This remains useful as a fallback for growth-like rows.

      - substrate_control_on
          Explicit scheduler permission for substrate temperature control.
          If false, the heater falls back to idle target behavior.

      - explicit_target_K
          Explicit recipe target for the substrate.
          This allows beam-off timed phases to hold a real elevated target
          even when deposition flux is zero.

      Control policy:
      - If job_active is false, target returns to idle baseline.
      - If substrate_control_on is false, target returns to idle baseline.
      - If an explicit non-idle target is supplied, that target is used.
      - Otherwise the heater falls back to the legacy flux-derived target.

      Fault-reset policy:
      Per-job miss streaks are reset only when the controlling job identity
      changes or when the active state changes. Scheduler transitions within
      the same controlling job should not reset fault history.
  */
  void setJobState(int job_index,
                   bool job_active,
                   double raw_job_flux_cm2s,
                   bool substrate_control_on,
                   double explicit_target_K);

  // Compute how much substrate-heater power is requested this tick.
  double computePowerRequestW();

  // Apply delivered heater power over dt seconds.
  void applyHeat(double watts, double dt_s);

  /*
      Backward-compatible one-sided readiness check.

      Existing behavior is preserved:
      - If there is no meaningful target, returns true.
      - Otherwise returns true once the substrate reaches the lower readiness band.

      This function does not distinguish "too hot" from "ready".
      New scheduler logic should prefer getThermalBandState().
  */
  bool isAtTarget() const;

  // Returns true when a meaningful non-idle substrate target exists.
  bool hasMeaningfulTarget() const;

  /*
      Scheduler-grade thermal-band query.

      lower_band_K
          Absolute acceptable margin below target.

      upper_band_K
          Absolute acceptable margin above target.

      Semantics:
      - Idle target                           -> Idle
      - T_sub < target - lower_band_K         -> BelowTargetBand
      - target-lower <= T_sub <= target+upper -> WithinTargetBand
      - T_sub > target + upper_band_K         -> AboveTargetBand
  */
  ThermalBandState getThermalBandState(double lower_band_K = READY_BAND_K_,
                                       double upper_band_K = READY_BAND_K_) const;

  // Convenience helpers for clearer scheduler code.
  bool isBelowTargetBand(double lower_band_K = READY_BAND_K_) const;
  bool isWithinTargetBand(double lower_band_K = READY_BAND_K_,
                          double upper_band_K = READY_BAND_K_) const;
  bool isAboveTargetBand(double upper_band_K = READY_BAND_K_) const;

  bool jobFailed() const { return job_failed_; }

  // Diagnostics accessors.
  double substrateTempK() const { return T_sub_K_; }
  double targetTempK() const { return T_target_K_; }
  double requestedPowerW() const { return P_requested_W_; }
  double deliveredPowerW() const { return P_delivered_W_; }
  int tempMissStreak() const { return temp_miss_streak_; }
  bool jobActive() const { return job_active_; }
  bool substrateControlOn() const { return substrate_control_on_; }

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
  int  job_index_              = -1;
  bool job_active_             = false;
  bool substrate_control_on_   = false;
  int  temp_miss_streak_       = 0;
  bool job_failed_             = false;
  bool is_leader_              = false;

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
  // This is used as a fallback for growth-like rows when no explicit target
  // is provided.
  double fluxToTargetTemp(double raw_job_flux_cm2s) const;

  // Estimate total thermal losses at a given substrate temperature.
  double lossPowerW(double T_K) const;
};