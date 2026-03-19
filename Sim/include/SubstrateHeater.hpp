#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Subsystem.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"

/*
    SubstrateHeater models the wafer-side thermal plant for the simulator.

    Design intent

    This class is responsible for the thermal state of the substrate only.
    It does not decide scheduler ownership, job advancement, or failure policy.
    Instead, it exposes a clean thermal interface that the scheduler can use.

    Core responsibilities

    1. Maintain the substrate temperature state.
    2. Accept scheduler-facing recipe intent for the active job.
    3. Estimate heater power needed to hold or approach the target.
    4. Evolve the thermal state after delivered heater power is known.
    5. Report readiness through a thermal-band interface.
    6. Track a substrate miss streak when execution-time monitoring is armed.
    7. Log the thermal environment seen by the wafer, including orbit-aware
       effective ambient temperature and absorbed solar power.

    Orbit-aware thermal model

    This class supports a lightweight orbit-aware environment model driven by
    a solar-scale signal from the orbit subsystem.

    The effective ambient temperature is modeled as

        T_env_eff = T_night + (T_day - T_night) * solar_scale

    The absorbed solar heating term is modeled as

        P_solar_abs = alpha_abs * A_proj * G_solar * solar_scale

    where solar_scale is expected to lie in the range [0, 1].

    Thermal exchange semantics

    Thermal exchange with the environment is modeled as a signed quantity.

        P_env_exchange(T) =
            emissivity * sigma * area * (T^4 - T_env_eff^4)
          + h_cond * (T - T_env_eff)

    Interpretation

    - Positive P_env_exchange means net heat leaves the substrate.
    - Negative P_env_exchange means the effective environment is passively
      warming the substrate.

    Scheduler contract

    The scheduler tells this class whether a meaningful substrate target is
    active for the current controlling phase. When there is no active target,
    the heater returns to idle behavior and readiness queries treat the
    substrate as non-blocking.

    Important semantic note

    The logged T_env_eff_K value represents the effective ambient or
    environment temperature the wafer is experiencing during that tick under
    the orbit-aware solar-scale model.
*/
class SubstrateHeater : public Subsystem {
public:
  /*
      Scheduler-facing thermal-band state.

      Why this exists

      The scheduler should be able to distinguish four thermal situations:

      Idle
          No meaningful substrate target is active, so the substrate should
          not block execution.

      BelowTargetBand
          The substrate is colder than the acceptable lower bound and still
          requires heating before the current phase is thermally ready.

      WithinTargetBand
          The substrate is inside the acceptable operating window for the
          current recipe phase.

      AboveTargetBand
          The substrate is hotter than the acceptable upper bound and must
          cool before the current phase is thermally ready.
  */
  enum class ThermalBandState {
    Idle = 0,
    BelowTargetBand,
    WithinTargetBand,
    AboveTargetBand
  };

  /*
      Construct the substrate thermal subsystem.

      maxPower_W
          Maximum heater power that may be requested by this subsystem.

      wafer_radius_m
          Radius of the wafer in meters. This is used to estimate wafer area
          for radiative loss and a simple projected solar-loading area.
  */
  explicit SubstrateHeater(double maxPower_W = 3000.0,
                           double wafer_radius_m = 0.15);

  /*
      Only the leader rank should emit CSV rows.

      This avoids duplicate logging in MPI runs where the same thermal state
      update logic executes on multiple ranks.
  */
  void setIsLeader(bool isLeader) { is_leader_ = isLeader; }

  /*
      Update the orbit-aware thermal environment using the current solar scale.

      solar_scale
          Orbit-driven sunlight factor expected to range from 0 for eclipse
          to 1 for full sun exposure.

      This updates two internal quantities:

      1. T_env_eff_K_
         The effective ambient temperature currently surrounding the wafer.

      2. P_solar_abs_W_
         The solar power absorbed directly by the wafer-side thermal node.

      This method does not itself advance temperature. It only updates the
      environment against which the next power request and thermal update
      will be computed.
  */
  void setOrbitThermalEnvironment(double solar_scale);

  /*
      Update the current scheduler-facing substrate control state.

      job_index
          Identity of the currently controlling job.

      job_active
          True when an active controlling row owns substrate behavior during
          this tick.

      raw_job_flux_cm2s
          Physical deposition flux request from the current schedule row.
          This is used as a fallback way to infer a substrate target when
          no explicit target is supplied.

      substrate_control_on
          Explicit scheduler permission for substrate temperature control.
          If false, the heater returns to idle-target behavior.

      explicit_target_K
          Explicit substrate target requested by the recipe phase.
          This allows non-growth timed phases to hold a real elevated wafer
          temperature even when deposition flux is zero.

      Control policy

      - If no job is active, target returns to idle baseline.
      - If substrate control is disabled, target returns to idle baseline.
      - If a meaningful explicit target is supplied, that target is used.
      - Otherwise the target falls back to the legacy flux-derived target.

      Fault-reset policy

      Miss-streak state is reset only when the controlling job identity
      changes or the active-state flag changes. Scheduler transitions inside
      the same controlling job should not wipe fault history.
  */
  void setJobState(int job_index,
                   bool job_active,
                   double raw_job_flux_cm2s,
                   bool substrate_control_on,
                   double explicit_target_K);

  /*
      Arm or disarm substrate failure monitoring.

      When armed, the subsystem counts consecutive ticks where the substrate
      is below the lower readiness band. When the streak reaches the configured
      failure limit, job_failed_ is latched true.

      When disarmed, the miss streak is reset immediately.
  */
  void setFailureMonitorArmed(bool armed);

  /*
      Compute the heater power request for the current tick.

      This uses a lightweight control law with two pieces:

      1. Feed-forward hold power
         Estimates the heater power needed to hold the target against the
         current signed environment-exchange term, minus any absorbed solar
         heating already available.

      2. Proportional heating term
         Adds extra demand when the substrate is still below target.

      Returns the requested heater power in watts, already clamped to the
      subsystem maximum.
  */
  double computePowerRequestW();

  /*
      Apply delivered heater power over the current time step.

      watts
          Heater power actually delivered by the power system.

      dt_s
          Time step in seconds.

      Net thermal power is modeled as

          P_net = P_delivered + P_solar_abs - P_env_exchange

      where P_env_exchange is signed:

      - positive when heat leaves the substrate
      - negative when the environment is passively warming the substrate

      The substrate temperature is advanced using the lumped thermal
      capacitance C_J_per_K_.
  */
  void applyHeat(double watts, double dt_s);

  /*
      Backward-compatible readiness check.

      Existing one-sided semantics are intentionally preserved:

      - If no meaningful target exists, returns true.
      - Otherwise returns true once the substrate reaches the lower
        readiness threshold.

      This function does not distinguish between "ready" and "too hot".
      Scheduler logic that needs full state should prefer
      getThermalBandState().
  */
  bool isAtTarget() const;

  /*
      Returns true when a meaningful non-idle substrate target is active.

      This is useful for distinguishing real recipe-controlled behavior from
      idle fallback behavior.
  */
  bool hasMeaningfulTarget() const;

  /*
      Query the full thermal-band state.

      lower_band_K
          Allowed margin below target before the substrate is considered too
          cold.

      upper_band_K
          Allowed margin above target before the substrate is considered too
          hot.

      Semantics

      - No meaningful target                       -> Idle
      - T_sub < target - lower_band_K              -> BelowTargetBand
      - target-lower <= T_sub <= target+upper      -> WithinTargetBand
      - T_sub > target + upper_band_K              -> AboveTargetBand
  */
  ThermalBandState getThermalBandState(double lower_band_K = READY_BAND_K_,
                                       double upper_band_K = READY_BAND_K_) const;

  /*
      Convenience helpers that make scheduler code easier to read.
  */
  bool isBelowTargetBand(double lower_band_K = READY_BAND_K_) const;
  bool isWithinTargetBand(double lower_band_K = READY_BAND_K_,
                          double upper_band_K = READY_BAND_K_) const;
  bool isAboveTargetBand(double upper_band_K = READY_BAND_K_) const;

  /*
      Latched execution-time failure result for the currently controlled job.
  */
  bool jobFailed() const { return job_failed_; }

  /*
      Diagnostics accessors.

      These provide direct visibility into the thermal state, control request,
      orbit-aware effective environment, and failure-monitoring state.
  */
  double substrateTempK() const { return T_sub_K_; }
  double targetTempK() const { return T_target_K_; }
  double ambientTempK() const { return T_env_eff_K_; }
  double solarScale() const { return solar_scale_; }
  double solarAbsorbedPowerW() const { return P_solar_abs_W_; }
  double requestedPowerW() const { return P_requested_W_; }
  double deliveredPowerW() const { return P_delivered_W_; }
  int tempMissStreak() const { return temp_miss_streak_; }
  bool jobActive() const { return job_active_; }
  bool substrateControlOn() const { return substrate_control_on_; }

  void initialize() override;
  void tick(const TickContext& ctx) override;
  void shutdown() override;

private:
  /*
      Geometry and hard limits.
  */
  double wafer_radius_m_;
  double wafer_area_m2_;
  double maxPower_W_;

  /*
      Thermal state.

      T_sub_K_
          Current substrate temperature.

      T_target_K_
          Current scheduler-facing substrate target.

      T_env_K_
          Legacy environment field retained for compatibility.

      T_env_eff_K_
          Effective ambient temperature seen by the wafer under the
          orbit-aware thermal model.

      solar_scale_
          Current orbit-driven sunlight factor.

      P_solar_abs_W_
          Current absorbed solar heating power applied directly to the
          substrate thermal node.
  */
  double T_sub_K_       = 300.0;
  double T_target_K_    = 300.0;
  double T_env_K_       = 300.0;
  double T_env_eff_K_   = 300.0;
  double solar_scale_   = 1.0;
  double P_solar_abs_W_ = 0.0;

  /*
      Power bookkeeping.

      P_requested_W_
          Power requested by the control law.

      P_delivered_W_
          Power actually delivered by the power system.

      last_P_loss_W_
          Latest signed environment-exchange term evaluated during the state
          update.

          Interpretation:
          - positive means net heat left the substrate
          - negative means the environment passively warmed the substrate

          The member name and CSV field name are retained for compatibility.
  */
  double P_requested_W_ = 0.0;
  double P_delivered_W_ = 0.0;
  double last_P_loss_W_ = 0.0;

  /*
      Scheduler-facing job state and failure tracking.
  */
  int  job_index_             = -1;
  bool job_active_            = false;
  bool substrate_control_on_  = false;
  int  temp_miss_streak_      = 0;
  bool job_failed_            = false;
  bool is_leader_             = false;
  bool failure_monitor_armed_ = false;

  /*
      Physical constants for the lightweight substrate model.

      sigma_
          Stefan-Boltzmann constant for radiative exchange.

      emissivity_
          Effective wafer-side emissivity.

      h_cond_WK_
          Lumped linear conductive or parasitic exchange term.

      C_J_per_K_
          Lumped thermal capacitance of the substrate node.

      Kp_W_per_K_
          Proportional controller gain.
  */
  static constexpr double sigma_      = 5.670374419e-8;
  static constexpr double emissivity_ = 0.80;
  static constexpr double h_cond_WK_  = 0.50;
  static constexpr double C_J_per_K_  = 1500.0;
  static constexpr double Kp_W_per_K_ = 25.0;

  /*
      Orbit-aware thermal environment constants.

      T_env_night_K_
          Effective ambient temperature during eclipse-like conditions.

      T_env_day_K_
          Effective ambient temperature during sunlit conditions.

      alpha_abs_
          Effective absorptivity for direct solar loading.

      A_proj_m2_
          Simple projected area used for absorbed solar power.

      G_solar_W_m2_
          Solar constant used by the lightweight model.
  */
  double T_env_night_K_ = 250.0;
  double T_env_day_K_   = 320.0;
  double alpha_abs_     = 0.30;
  double A_proj_m2_     = 0.0;
  double G_solar_W_m2_  = 1361.0;

  /*
      Gate and failure thresholds.

      READY_BAND_K_
          Default absolute readiness band around target.

      FAIL_LIMIT_TICKS_
          Number of consecutive below-band ticks required to latch a failure
          once monitoring is armed.
  */
  static constexpr double READY_BAND_K_     = 5.0;
  static constexpr int    FAIL_LIMIT_TICKS_ = 20;

  /*
      Map physical deposition flux to a fallback substrate target temperature.

      This is used when the scheduler has enabled substrate control but the
      current phase did not provide an explicit elevated target.
  */
  double fluxToTargetTemp(double raw_job_flux_cm2s) const;

  /*
      Estimate the signed net thermal exchange at a given substrate
      temperature under the current effective environment.

      Exchange model

      1. Radiative exchange against T_env_eff_K_
      2. Linear conductive or parasitic exchange against T_env_eff_K_

      Sign convention

      - Positive return value means heat leaves the substrate.
      - Negative return value means the environment is warming the substrate.
  */
  double lossPowerW(double T_K) const;

  /*
      Compute the current effective ambient temperature from solar scale.
  */
  double computeEffectiveEnvTempK() const;

  /*
      Compute the current absorbed solar power from solar scale.
  */
  double computeSolarAbsorbedPowerW() const;
};