#include "SubstrateHeater.hpp"

#include <algorithm>
#include <cmath>

/*
    SubstrateHeater

    This file owns the wafer thermal state.

    Runtime configuration

    C_J, emissivity, ready band, and failure limit are configured through the
    inline configureThermalModel method in SubstrateHeater.hpp.

    The maximum substrate heater power is configured through the constructor
    argument passed by main.cpp.

    Thermal model

    P_env_exchange =
        emissivity * sigma * area * (T_sub^4 minus T_env_eff^4)
        plus h_cond * (T_sub minus T_env_eff)

    P_net =
        P_delivered + P_solar_abs minus P_env_exchange

    dT =
        P_net / C_J * dt

    Sign convention

    Positive P_env_exchange means heat leaves the substrate.

    Negative P_env_exchange means the environment warms the substrate.
*/

namespace {

/*
    Local constants for the substrate thermal subsystem.
*/
constexpr double kPi = 3.14159265358979323846;
constexpr double kIdleTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;

}  // namespace

SubstrateHeater::SubstrateHeater(double maxPower_W, double wafer_radius_m)
    : Subsystem("substrate"),
      wafer_radius_m_((std::isfinite(wafer_radius_m) && wafer_radius_m > 0.0)
                          ? wafer_radius_m
                          : 0.15),
      wafer_area_m2_(kPi * wafer_radius_m_ * wafer_radius_m_),
      maxPower_W_((std::isfinite(maxPower_W) && maxPower_W > 0.0)
                      ? maxPower_W
                      : 3000.0) {
  /*
      Use wafer frontal area as the projected area for the absorbed solar term.

      This keeps the thermal model lightweight while remaining interpretable.
  */
  A_proj_m2_ = wafer_area_m2_;
}

void SubstrateHeater::initialize() {
  /*
      Reset dynamic thermal state.

      Runtime configured constants are intentionally not reset here.
  */
  T_sub_K_       = kIdleTempK;
  T_target_K_    = kIdleTempK;
  T_env_K_       = kIdleTempK;
  T_env_eff_K_   = kIdleTempK;
  solar_scale_   = 1.0;
  P_solar_abs_W_ = 0.0;

  /*
      Reset power bookkeeping.
  */
  P_requested_W_ = 0.0;
  P_delivered_W_ = 0.0;
  last_P_loss_W_ = 0.0;

  /*
      Reset scheduler facing job state and failure monitoring.
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
      No owned external resources require shutdown.
  */
}

void SubstrateHeater::setOrbitThermalEnvironment(double solar_scale) {
  /*
      Clamp the orbit driven solar scale to a safe range.
  */
  double s = solar_scale;
  if (!std::isfinite(s)) {
    s = 0.0;
  }

  solar_scale_ = std::clamp(s, 0.0, 1.0);

  /*
      Update the effective environment used by both the control law and the
      state update.
  */
  T_env_eff_K_   = computeEffectiveEnvTempK();
  P_solar_abs_W_ = computeSolarAbsorbedPowerW();

  /*
      Keep the legacy environment field synchronized for compatibility.
  */
  T_env_K_ = T_env_eff_K_;
}

double SubstrateHeater::computeEffectiveEnvTempK() const {
  /*
      Interpolate between eclipse and sunlit effective ambient temperatures.
  */
  return T_env_night_K_ +
         (T_env_day_K_ - T_env_night_K_) * solar_scale_;
}

double SubstrateHeater::computeSolarAbsorbedPowerW() const {
  /*
      Direct absorbed solar heating term.
  */
  const double p =
      alpha_abs_ * A_proj_m2_ * G_solar_W_m2_ * solar_scale_;

  if (!std::isfinite(p) || p < 0.0) {
    return 0.0;
  }

  return p;
}

void SubstrateHeater::setJobState(int job_index,
                                  bool job_active,
                                  double raw_job_flux_cm2s,
                                  bool substrate_control_on,
                                  double explicit_target_K) {
  /*
      Reset fault history only when job ownership or active state changes.

      This preserves the failure streak across internal state transitions of
      the same controlling job.
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
      No active controlling job means idle substrate behavior.
  */
  if (!job_active_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  /*
      If this recipe phase does not control the substrate, return to idle.
  */
  if (!substrate_control_on_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  /*
      Prefer an explicit elevated recipe target when supplied.
  */
  if (std::isfinite(explicit_target_K) &&
      explicit_target_K > (kIdleTempK + kMeaningfulTargetMarginK)) {
    T_target_K_ = explicit_target_K;
    return;
  }

  /*
      Fall back to the legacy flux derived target if no explicit target is
      available.
  */
  T_target_K_ = fluxToTargetTemp(raw_job_flux_cm2s);
}

void SubstrateHeater::setFailureMonitorArmed(bool armed) {
  /*
      Enable or disable execution time failure monitoring.
  */
  failure_monitor_armed_ = armed;

  /*
      Disarming clears the streak so the next monitoring window starts cleanly.
  */
  if (!failure_monitor_armed_) {
    temp_miss_streak_ = 0;
  }
}

double SubstrateHeater::fluxToTargetTemp(double raw_job_flux_cm2s) const {
  /*
      Convert deposition flux to a fallback substrate target.

      This is a scheduler level approximation. Explicit recipe targets should
      be preferred when present.
  */
  if (!std::isfinite(raw_job_flux_cm2s) ||
      raw_job_flux_cm2s <= 0.0) {
    return kIdleTempK;
  }

  const double F_low  = 2.0e12;
  const double F_high = 9.0e13;
  const double T_low  = 700.0;
  const double T_high = 850.0;

  const double F_clamped =
      std::clamp(raw_job_flux_cm2s, F_low, F_high);

  const double denom =
      std::log(F_high) - std::log(F_low);

  if (!std::isfinite(denom) || denom <= 0.0) {
    return T_low;
  }

  const double alpha =
      (std::log(F_clamped) - std::log(F_low)) / denom;

  return T_low + std::clamp(alpha, 0.0, 1.0) * (T_high - T_low);
}

double SubstrateHeater::lossPowerW(double T_K) const {
  /*
      Compute signed environment exchange at a proposed substrate temperature.
  */
  if (!std::isfinite(T_K)) {
    return 0.0;
  }

  const double T_env =
      std::isfinite(T_env_eff_K_) ? T_env_eff_K_ : kIdleTempK;

  const double P_rad =
      emissivity_ *
      sigma_ *
      wafer_area_m2_ *
      (std::pow(T_K, 4) - std::pow(T_env, 4));

  const double P_cond =
      h_cond_WK_ * (T_K - T_env);

  const double total = P_rad + P_cond;

  if (!std::isfinite(total)) {
    return 0.0;
  }

  return total;
}

double SubstrateHeater::computePowerRequestW() {
  /*
      Without a meaningful active target, the substrate requests no heater
      power.
  */
  if (!job_active_ ||
      !substrate_control_on_ ||
      T_target_K_ <= (kIdleTempK + kMeaningfulTargetMarginK)) {
    P_requested_W_ = 0.0;
    return 0.0;
  }

  const double err_K = T_target_K_ - T_sub_K_;

  /*
      Feed forward term.

      Estimate the power required to hold the target against environment
      exchange, then subtract absorbed sunlight already entering the wafer.
  */
  const double P_ff =
      std::max(0.0, lossPowerW(T_target_K_) - P_solar_abs_W_);

  /*
      Proportional term.

      Add extra heating only when the substrate is below target.
  */
  const double P_p =
      (err_K > 0.0) ? (Kp_W_per_K_ * err_K) : 0.0;

  const double request =
      P_ff + P_p;

  if (!std::isfinite(request)) {
    P_requested_W_ = 0.0;
    return 0.0;
  }

  P_requested_W_ = std::clamp(request, 0.0, maxPower_W_);
  return P_requested_W_;
}

void SubstrateHeater::applyHeat(double watts, double dt_s) {
  /*
      Sanitize time step and delivered heater power.
  */
  const double dt =
      (std::isfinite(dt_s) && dt_s > 0.0) ? dt_s : 0.0;

  P_delivered_W_ =
      (std::isfinite(watts) && watts > 0.0) ? watts : 0.0;

  /*
      Evaluate signed environment exchange at the current substrate
      temperature.
  */
  last_P_loss_W_ = lossPowerW(T_sub_K_);

  /*
      Protect against invalid configured thermal capacitance.
  */
  const double effective_C_J =
      (std::isfinite(C_J_per_K_) && C_J_per_K_ > 0.0)
          ? C_J_per_K_
          : 1500.0;

  /*
      Compute net power into the substrate node.
  */
  const double net_W =
      P_delivered_W_ + P_solar_abs_W_ - last_P_loss_W_;

  /*
      Advance the substrate temperature.
  */
  T_sub_K_ += (net_W / effective_C_J) * dt;

  /*
      Repair invalid or unphysical states.
  */
  if (!std::isfinite(T_sub_K_)) {
    T_sub_K_ = T_env_eff_K_;
  }

  if (T_sub_K_ < 0.0) {
    T_sub_K_ = 0.0;
  }
}

bool SubstrateHeater::hasMeaningfulTarget() const {
  /*
      A target is meaningful only when a controlling job has enabled substrate
      thermal control and the target is clearly above idle.
  */
  return job_active_ &&
         substrate_control_on_ &&
         std::isfinite(T_target_K_) &&
         (T_target_K_ > (kIdleTempK + kMeaningfulTargetMarginK));
}

bool SubstrateHeater::isAtTarget() const {
  /*
      Preserve old one sided readiness semantics.
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
      Idle targets should not block scheduler progress.
  */
  if (!hasMeaningfulTarget()) {
    return ThermalBandState::Idle;
  }

  /*
      Invalid state is treated conservatively as too cold.
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
  /*
      Ask the full classifier for the below target state.
  */
  return getThermalBandState(lower_band_K, READY_BAND_K_) ==
         ThermalBandState::BelowTargetBand;
}

bool SubstrateHeater::isWithinTargetBand(double lower_band_K,
                                         double upper_band_K) const {
  /*
      Ask the full classifier for the ready state.
  */
  return getThermalBandState(lower_band_K, upper_band_K) ==
         ThermalBandState::WithinTargetBand;
}

bool SubstrateHeater::isAboveTargetBand(double upper_band_K) const {
  /*
      Ask the full classifier for the too hot state.
  */
  return getThermalBandState(READY_BAND_K_, upper_band_K) ==
         ThermalBandState::AboveTargetBand;
}

void SubstrateHeater::tick(const TickContext& ctx) {
  /*
      Execution time failure monitoring is separate from normal readiness.

      When armed, consecutive below band ticks count toward substrate failure.

      When disarmed, the streak is cleared.
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

  /*
      Latch substrate failure after the configured streak limit.
  */
  if (temp_miss_streak_ >= FAIL_LIMIT_TICKS_) {
    job_failed_ = true;
  }

  /*
      Only the leader writes substrate CSV rows.
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
            "h_WK",
            "ready_band_K",
            "fail_limit_ticks"
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
            h_cond_WK_,
            READY_BAND_K_,
            static_cast<double>(FAIL_LIMIT_TICKS_)
        }
    );
  }
}