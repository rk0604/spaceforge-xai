#include "SubstrateHeater.hpp"

namespace {

/*
    Basic local constants for the substrate thermal subsystem.

    kPi
        Used to compute wafer area from wafer radius.

    kIdleTempK
        Idle baseline temperature used whenever there is no meaningful
        scheduler-controlled substrate target.

    kMeaningfulTargetMarginK
        Minimum elevation above idle needed before a target is treated as
        a real thermal control target instead of an idle-like value.
*/
constexpr double kPi = 3.14159265358979323846;
constexpr double kIdleTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;

}  // namespace

SubstrateHeater::SubstrateHeater(double maxPower_W, double wafer_radius_m)
    : Subsystem("substrate"),
      wafer_radius_m_(wafer_radius_m),
      wafer_area_m2_(kPi * wafer_radius_m * wafer_radius_m),
      maxPower_W_(std::max(0.0, maxPower_W)) {
  /*
      Use wafer frontal area as a simple projected area approximation for the
      absorbed solar term. This keeps the model lightweight and tunable while
      remaining physically interpretable.
  */
  A_proj_m2_ = wafer_area_m2_;
}

void SubstrateHeater::initialize() {
  /*
      Reset thermal state to a clean idle baseline.

      The orbit-aware environment is also reset here so that standalone runs
      or early initialization steps begin from a stable state even before the
      first orbit update is pushed in.
  */
  T_sub_K_       = kIdleTempK;
  T_target_K_    = kIdleTempK;
  T_env_K_       = kIdleTempK;
  T_env_eff_K_   = kIdleTempK;
  solar_scale_   = 1.0;
  P_solar_abs_W_ = 0.0;

  /*
      Reset all power bookkeeping.
  */
  P_requested_W_ = 0.0;
  P_delivered_W_ = 0.0;
  last_P_loss_W_ = 0.0;

  /*
      Reset scheduler-facing job state and failure-monitoring state.
  */
  job_index_             = -1;
  job_active_            = false;
  substrate_control_on_  = false;
  temp_miss_streak_      = 0;
  job_failed_            = false;
  failure_monitor_armed_ = false;
}

void SubstrateHeater::shutdown() {
  /*
      No special shutdown action is required.

      The subsystem owns no external resources that need explicit release.
  */
}

void SubstrateHeater::setOrbitThermalEnvironment(double solar_scale) {
  /*
      Clamp the external solar scale to a safe and interpretable range.

      This prevents invalid values from producing unstable thermal behavior.
  */
  double s = solar_scale;
  if (!std::isfinite(s)) {
    s = 0.0;
  }
  solar_scale_ = std::clamp(s, 0.0, 1.0);

  /*
      Update the orbit-aware effective environment used by both the control
      law and the thermal state update.
  */
  T_env_eff_K_ = computeEffectiveEnvTempK();
  P_solar_abs_W_ = computeSolarAbsorbedPowerW();

  /*
      Keep the legacy environment field synchronized so any older code paths
      that still inspect T_env_K_ remain numerically consistent.
  */
  T_env_K_ = T_env_eff_K_;
}

double SubstrateHeater::computeEffectiveEnvTempK() const {
  /*
      Interpolate linearly between eclipse-like and sunlit ambient
      conditions using the current orbit-driven solar scale.
  */
  return T_env_night_K_ + (T_env_day_K_ - T_env_night_K_) * solar_scale_;
}

double SubstrateHeater::computeSolarAbsorbedPowerW() const {
  /*
      Lightweight direct solar heating model.

      This is intentionally simple:
      absorptivity times projected area times solar constant times sunlight
      fraction.
  */
  return alpha_abs_ * A_proj_m2_ * G_solar_W_m2_ * solar_scale_;
}

void SubstrateHeater::setJobState(int job_index,
                                  bool job_active,
                                  double raw_job_flux_cm2s,
                                  bool substrate_control_on,
                                  double explicit_target_K) {
  /*
      Reset fault history only when job ownership or overall active status
      changes. This preserves continuity across phase transitions within the
      same controlling job.
  */
  if (job_index != job_index_ || job_active != job_active_) {
    temp_miss_streak_      = 0;
    job_failed_            = false;
    failure_monitor_armed_ = false;
  }

  job_index_            = job_index;
  job_active_           = job_active;
  substrate_control_on_ = substrate_control_on;

  /*
      No active controlling job means the substrate should fall back to the
      idle baseline target.
  */
  if (!job_active_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  /*
      If the scheduler has disabled substrate control for this phase, the
      substrate target also returns to idle baseline.
  */
  if (!substrate_control_on_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  /*
      Prefer an explicit elevated recipe target when one is supplied.
      This is important for beam-off timed phases that still require real
      substrate thermal conditioning.
  */
  if (std::isfinite(explicit_target_K) &&
      explicit_target_K > (kIdleTempK + kMeaningfulTargetMarginK)) {
    T_target_K_ = explicit_target_K;
    return;
  }

  /*
      Fall back to the legacy flux-derived substrate target only when no
      meaningful explicit recipe target was supplied.
  */
  T_target_K_ = fluxToTargetTemp(raw_job_flux_cm2s);
}

void SubstrateHeater::setFailureMonitorArmed(bool armed) {
  failure_monitor_armed_ = armed;

  /*
      Disarming the execution-time failure monitor should also clear the
      running miss streak so that monitoring restarts cleanly next time.
  */
  if (!failure_monitor_armed_) {
    temp_miss_streak_ = 0;
  }
}

double SubstrateHeater::fluxToTargetTemp(double raw_job_flux_cm2s) const {
  /*
      Map physical deposition flux to a fallback substrate target.

      This is not intended to be a first-principles wafer process model.
      It is a lightweight monotonic schedule-aware mapping that preserves the
      simulator's recipe-driven behavior when explicit targets are absent.
  */
  if (!std::isfinite(raw_job_flux_cm2s) || raw_job_flux_cm2s <= 0.0) {
    return kIdleTempK;
  }

  const double F_low  = 2.0e12;
  const double F_high = 9.0e13;
  const double T_low  = 700.0;
  const double T_high = 850.0;

  const double F_clamped = std::clamp(raw_job_flux_cm2s, F_low, F_high);
  const double alpha =
      (std::log(F_clamped) - std::log(F_low)) /
      (std::log(F_high) - std::log(F_low));

  return T_low + std::clamp(alpha, 0.0, 1.0) * (T_high - T_low);
}

double SubstrateHeater::lossPowerW(double T_K) const {
  /*
      Compute total loss against the current orbit-aware effective
      environment.

      The loss model contains two terms:

      1. Radiative exchange
      2. Linear conductive or parasitic loss

      The max-with-zero structure keeps the model simple and prevents the loss
      term from acting like an active heater when the substrate is colder than
      the environment. External warming from the environment is instead
      represented through the environment temperature itself and the explicit
      absorbed solar term.
  */
  const double T_env = T_env_eff_K_;

  const double P_rad =
      emissivity_ * sigma_ * wafer_area_m2_ *
      (std::pow(T_K, 4) - std::pow(T_env, 4));

  const double P_cond = h_cond_WK_ * (T_K - T_env);

  return std::max(0.0, P_rad) + std::max(0.0, P_cond);
}

double SubstrateHeater::computePowerRequestW() {
  /*
      If there is no meaningful active substrate target, request no power.

      This preserves idle behavior and prevents the substrate controller from
      heating unnecessarily during non-controlling or substrate-off phases.
  */
  if (!job_active_ || !substrate_control_on_ ||
      T_target_K_ <= (kIdleTempK + kMeaningfulTargetMarginK)) {
    P_requested_W_ = 0.0;
    return 0.0;
  }

  const double err_K = T_target_K_ - T_sub_K_;

  /*
      Feed-forward term

      Estimate how much heater power would be needed to hold the target
      against current thermal losses under the current orbit-aware ambient
      conditions, then subtract absorbed solar heat that is already available.

      This prevents the controller from over-requesting heater power during
      sunlit portions of the orbit.
  */
  const double P_ff =
      std::max(0.0, lossPowerW(T_target_K_) - P_solar_abs_W_);

  /*
      Proportional term

      Add extra heating when the substrate is still below target.
      No active cooling term is introduced here. Cooling remains passive.
  */
  const double P_p = (err_K > 0.0) ? (Kp_W_per_K_ * err_K) : 0.0;

  P_requested_W_ = std::clamp(P_ff + P_p, 0.0, maxPower_W_);
  return P_requested_W_;
}

void SubstrateHeater::applyHeat(double watts, double dt_s) {
  const double dt = std::max(0.0, dt_s);

  /*
      Record actual delivered heater power for diagnostics and logging.
  */
  P_delivered_W_ = std::max(0.0, watts);

  /*
      Evaluate losses at the current substrate temperature against the current
      orbit-aware effective environment.
  */
  last_P_loss_W_ = lossPowerW(T_sub_K_);

  /*
      Net power into the substrate node.

      Positive contributions
      - delivered heater power
      - absorbed direct solar power

      Negative contribution
      - total thermal loss
  */
  const double net_W = P_delivered_W_ + P_solar_abs_W_ - last_P_loss_W_;

  /*
      Advance the substrate temperature using the lumped thermal capacitance.
  */
  T_sub_K_ += (net_W / C_J_per_K_) * dt;

  /*
      Clamp invalid or unphysical states back to a safe baseline.
  */
  if (!std::isfinite(T_sub_K_)) {
    T_sub_K_ = T_env_eff_K_;
  }

  T_sub_K_ = std::max(0.0, T_sub_K_);
}

bool SubstrateHeater::hasMeaningfulTarget() const {
  /*
      A target is meaningful only when the scheduler has both activated the
      job and enabled substrate control, and the target is clearly above the
      idle baseline.
  */
  return job_active_ &&
         substrate_control_on_ &&
         std::isfinite(T_target_K_) &&
         (T_target_K_ > (kIdleTempK + kMeaningfulTargetMarginK));
}

bool SubstrateHeater::isAtTarget() const {
  /*
      Preserve legacy one-sided readiness semantics.

      If no meaningful target exists, the substrate should not block
      scheduler progress. Otherwise readiness is reached once the substrate
      temperature crosses the lower readiness threshold.
  */
  if (!hasMeaningfulTarget()) {
    return true;
  }

  return T_sub_K_ >= (T_target_K_ - READY_BAND_K_);
}

SubstrateHeater::ThermalBandState
SubstrateHeater::getThermalBandState(double lower_band_K,
                                     double upper_band_K) const {
  /*
      No meaningful target means the substrate is effectively idle from the
      scheduler's perspective.
  */
  if (!hasMeaningfulTarget()) {
    return ThermalBandState::Idle;
  }

  /*
      Invalid thermal state is treated conservatively as below target.
  */
  if (!std::isfinite(T_sub_K_) || !std::isfinite(T_target_K_)) {
    return ThermalBandState::BelowTargetBand;
  }

  double lower = lower_band_K;
  double upper = upper_band_K;

  if (!std::isfinite(lower) || lower < 0.0) {
    lower = READY_BAND_K_;
  }
  if (!std::isfinite(upper) || upper < 0.0) {
    upper = READY_BAND_K_;
  }

  const double lower_bound_K = T_target_K_ - lower;
  const double upper_bound_K = T_target_K_ + upper;

  if (T_sub_K_ < lower_bound_K) {
    return ThermalBandState::BelowTargetBand;
  }
  if (T_sub_K_ > upper_bound_K) {
    return ThermalBandState::AboveTargetBand;
  }

  return ThermalBandState::WithinTargetBand;
}

bool SubstrateHeater::isBelowTargetBand(double lower_band_K) const {
  return getThermalBandState(lower_band_K, READY_BAND_K_) ==
         ThermalBandState::BelowTargetBand;
}

bool SubstrateHeater::isWithinTargetBand(double lower_band_K,
                                         double upper_band_K) const {
  return getThermalBandState(lower_band_K, upper_band_K) ==
         ThermalBandState::WithinTargetBand;
}

bool SubstrateHeater::isAboveTargetBand(double upper_band_K) const {
  return getThermalBandState(READY_BAND_K_, upper_band_K) ==
         ThermalBandState::AboveTargetBand;
}

void SubstrateHeater::tick(const TickContext& ctx) {
  /*
      Execution-time failure monitoring is intentionally separate from the
      normal readiness query.

      When monitoring is armed, count consecutive ticks where the substrate
      is below the lower readiness band. When the streak crosses the failure
      threshold, latch job_failed_ true.

      When monitoring is not armed, clear the streak.
  */
  if (failure_monitor_armed_ && hasMeaningfulTarget()) {
    if (T_sub_K_ < (T_target_K_ - READY_BAND_K_)) {
      temp_miss_streak_ += 1;
    } else {
      temp_miss_streak_ = 0;
    }
  } else {
    temp_miss_streak_ = 0;
  }

  if (temp_miss_streak_ >= FAIL_LIMIT_TICKS_) {
    job_failed_ = true;
  }

  /*
      Only the leader writes CSV rows.

      Important logging note

      T_env_eff_K is the effective ambient or environment temperature the
      wafer is experiencing during this tick under the orbit-aware solar-scale
      thermal model.
  */
  if (is_leader_) {
    Logger::instance().log_wide(
        name_,
        ctx.tick_index,
        ctx.time,
        {
            "job_index",
            "job_active",
            "substrate_control_on",
            "T_sub_K",
            "T_target_K",
            "T_env_eff_K",
            "solar_scale",
            "P_solar_abs_W",
            "P_req_W",
            "P_deliv_W",
            "P_loss_W",
            "streak",
            "failed",
            "C_J",
            "eps",
            "h_WK"
        },
        {
            static_cast<double>(job_index_),
            job_active_ ? 1.0 : 0.0,
            substrate_control_on_ ? 1.0 : 0.0,
            T_sub_K_,
            T_target_K_,
            T_env_eff_K_,
            solar_scale_,
            P_solar_abs_W_,
            P_requested_W_,
            P_delivered_W_,
            last_P_loss_W_,
            static_cast<double>(temp_miss_streak_),
            job_failed_ ? 1.0 : 0.0,
            C_J_per_K_,
            emissivity_,
            h_cond_WK_
        });
  }
}