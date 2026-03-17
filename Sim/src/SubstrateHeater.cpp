#include "SubstrateHeater.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kIdleTempK = 300.0;
constexpr double kMeaningfulTargetMarginK = 10.0;
}

SubstrateHeater::SubstrateHeater(double maxPower_W, double wafer_radius_m)
: Subsystem("substrate"),
  wafer_radius_m_(wafer_radius_m),
  wafer_area_m2_(kPi * wafer_radius_m * wafer_radius_m),
  maxPower_W_(std::max(0.0, maxPower_W)) {}

void SubstrateHeater::initialize() {
  T_sub_K_                = kIdleTempK;
  T_target_K_             = kIdleTempK;
  T_env_K_                = kIdleTempK;

  P_requested_W_          = 0.0;
  P_delivered_W_          = 0.0;
  last_P_loss_W_          = 0.0;

  job_index_              = -1;
  job_active_             = false;
  substrate_control_on_   = false;
  temp_miss_streak_       = 0;
  job_failed_             = false;
  failure_monitor_armed_   = false;
}

void SubstrateHeater::shutdown() {}

void SubstrateHeater::setJobState(int job_index,
                                  bool job_active,
                                  double raw_job_flux_cm2s,
                                  bool substrate_control_on,
                                  double explicit_target_K) {
  // Reset per-job fault tracking only when the controlling job identity or
  // active status changes. Transitions inside the same controlling job should
  // preserve fault history.
  if (job_index != job_index_ || job_active != job_active_) {
    temp_miss_streak_      = 0;
    job_failed_            = false;
    failure_monitor_armed_ = false;
  }

  job_index_            = job_index;
  job_active_           = job_active;
  substrate_control_on_ = substrate_control_on;

  // No active controller means idle target.
  if (!job_active_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  // Explicitly disabled substrate control means idle target.
  if (!substrate_control_on_) {
    T_target_K_ = kIdleTempK;
    return;
  }

  // Prefer explicit recipe targets whenever a meaningful one is supplied.
  if (std::isfinite(explicit_target_K) &&
      explicit_target_K > (kIdleTempK + kMeaningfulTargetMarginK)) {
    T_target_K_ = explicit_target_K;
    return;
  }

  // Fallback to the legacy growth-like flux mapping when no explicit target is
  // available. This preserves old behavior for deposition-oriented rows.
  T_target_K_ = fluxToTargetTemp(raw_job_flux_cm2s);
}

void SubstrateHeater::setFailureMonitorArmed(bool armed) {
  failure_monitor_armed_ = armed;

  if (!failure_monitor_armed_) {
    temp_miss_streak_ = 0;
  }
}

double SubstrateHeater::fluxToTargetTemp(double raw_job_flux_cm2s) const {
  // No physical deposition request means no growth-like substrate target.
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
  const double Prad =
      emissivity_ * sigma_ * wafer_area_m2_ *
      (std::pow(T_K, 4) - std::pow(T_env_K_, 4));

  const double Pcond = h_cond_WK_ * (T_K - T_env_K_);

  return std::max(0.0, Prad) + std::max(0.0, Pcond);
}

double SubstrateHeater::computePowerRequestW() {
  // If there is no meaningful substrate target for this phase, request no heat.
  // This safely handles IDLE, SOURCE_DEGAS, and any substrate_on = 0 phases.
  if (!job_active_ || !substrate_control_on_ ||
      T_target_K_ <= (kIdleTempK + kMeaningfulTargetMarginK)) {
    P_requested_W_ = 0.0;
    return 0.0;
  }

  const double err_K = (T_target_K_ - T_sub_K_);

  // Feed-forward calculates the power needed to maintain the TARGET temperature.
  // This is the critical fix: if the target drops, the requested baseline power 
  // drops immediately, allowing the physical wafer to shed excess heat to space.
  const double P_ff = lossPowerW(T_target_K_);

  // Proportional heating provides a temporary boost only when below target.
  // There is no active cooling path; cooldown relies on P_ff being lower than current loss.
  const double P_p = (err_K > 0.0) ? (Kp_W_per_K_ * err_K) : 0.0;

  P_requested_W_ = std::clamp(P_ff + P_p, 0.0, maxPower_W_);
  return P_requested_W_;
}

  const double err_K = (T_target_K_ - T_sub_K_);

  // Feed-forward term offsets current thermal loss.
  const double P_ff = lossPowerW(T_sub_K_);

  // Simple proportional heating only when below target.
  // There is no active cooling path here. Cooldown remains passive and is
  // interpreted by the scheduler through thermal-band helpers.
  const double P_p = (err_K > 0.0) ? (Kp_W_per_K_ * err_K) : 0.0;

  P_requested_W_ = std::clamp(P_ff + P_p, 0.0, maxPower_W_);
  return P_requested_W_;
}

void SubstrateHeater::applyHeat(double watts, double dt_s) {
  const double dt = std::max(0.0, dt_s);

  P_delivered_W_ = std::max(0.0, watts);
  last_P_loss_W_ = lossPowerW(T_sub_K_);

  const double net_W = P_delivered_W_ - last_P_loss_W_;
  T_sub_K_ += (net_W / C_J_per_K_) * dt;

  if (!std::isfinite(T_sub_K_)) {
    T_sub_K_ = T_env_K_;
  }

  T_sub_K_ = std::max(0.0, T_sub_K_);
}

bool SubstrateHeater::hasMeaningfulTarget() const {
  return job_active_ &&
         substrate_control_on_ &&
         std::isfinite(T_target_K_) &&
         (T_target_K_ > (kIdleTempK + kMeaningfulTargetMarginK));
}

bool SubstrateHeater::isAtTarget() const {
  // Preserve the legacy one-sided readiness behavior for existing callers.
  if (!hasMeaningfulTarget()) {
    return true;
  }

  return (T_sub_K_ >= (T_target_K_ - READY_BAND_K_));
}

SubstrateHeater::ThermalBandState
SubstrateHeater::getThermalBandState(double lower_band_K,
                                     double upper_band_K) const {
  // Idle or non-meaningful targets should not block scheduler readiness.
  if (!hasMeaningfulTarget()) {
    return ThermalBandState::Idle;
  }

  if (!std::isfinite(T_sub_K_) || !std::isfinite(T_target_K_)) {
    // Conservative choice. Invalid thermal state should not be treated as ready.
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
  // Only accumulate misses when:
  // 1) this job actually has a meaningful substrate target, and
  // 2) the scheduler has armed execution-time failure monitoring.
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
        P_requested_W_,
        P_delivered_W_,
        last_P_loss_W_,
        static_cast<double>(temp_miss_streak_),
        job_failed_ ? 1.0 : 0.0,
        C_J_per_K_,
        emissivity_,
        h_cond_WK_
      }
    );
  }
}