// Sim/src/SubstrateHeater.cpp
#include "SubstrateHeater.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
}

SubstrateHeater::SubstrateHeater(double maxPower_W, double wafer_radius_m)
: Subsystem("substrate"),  // filename => substrate.csv
  wafer_radius_m_(wafer_radius_m),
  wafer_area_m2_(kPi * wafer_radius_m * wafer_radius_m),
  maxPower_W_(std::max(0.0, maxPower_W)) {}

void SubstrateHeater::initialize() {
  T_sub_K_          = 300.0;
  T_target_K_       = 300.0;
  T_env_K_          = 300.0;

  P_requested_W_    = 0.0;
  P_delivered_W_    = 0.0;

  job_index_        = -1;
  job_active_       = false;

  temp_miss_streak_ = 0;
  job_failed_       = false;

  // is_leader_ remains configured by setIsLeader(...)
}

void SubstrateHeater::shutdown() {
  // No-op; Logger creates/flushes per call to log_wide(...)
}

void SubstrateHeater::setJobState(int job_index, bool job_active, double Fwafer_cm2s) {
  // Reset gating/failure counters when job identity changes or active flag toggles
  if (job_index != job_index_ || job_active != job_active_) {
    temp_miss_streak_ = 0;
    job_failed_       = false;
  }

  job_index_  = job_index;
  job_active_ = job_active;

  if (job_active_) {
    T_target_K_ = fluxToTargetTemp(Fwafer_cm2s);
  } else {
    T_target_K_ = 300.0;
  }
}

double SubstrateHeater::fluxToTargetTemp(double F) const {
  // Log-interpolated mapping between anchor flux points.
  // Intended to produce substrate targets in the ~700–850 K range for typical growth fluxes.
  if (!std::isfinite(F) || F <= 0.0) return 300.0;

  const double F_low  = 5.0e13;
  const double F_high = 1.2e14;
  const double T_low  = 700.0;
  const double T_high = 850.0;

  const double logF     = std::log(F);
  const double logFlow  = std::log(F_low);
  const double logFhigh = std::log(F_high);
  const double denom    = (logFhigh - logFlow);

  double alpha = 0.0;
  if (std::isfinite(denom) && denom > 0.0) {
    alpha = (logF - logFlow) / denom;
  }
  alpha = std::clamp(alpha, 0.0, 1.0);

  return T_low + alpha * (T_high - T_low);
}

double SubstrateHeater::lossPowerW(double T_K) const {
  // P_loss(T) = P_rad(T) + P_cond(T), clamped to non-negative values.

  // Radiation: eps * sigma * A * (T^4 - T_env^4)
  const double Prad =
      emissivity_ * sigma_ * wafer_area_m2_ *
      (std::pow(T_K, 4) - std::pow(T_env_K_, 4));

  // Effective linear loss: h * (T - T_env)
  const double Pcond = h_cond_WK_ * (T_K - T_env_K_);

  const double Prad_pos  = (std::isfinite(Prad)  ? std::max(0.0, Prad)  : 0.0);
  const double Pcond_pos = (std::isfinite(Pcond) ? std::max(0.0, Pcond) : 0.0);

  return Prad_pos + Pcond_pos;
}

double SubstrateHeater::computePowerRequestW() {
  // Control law:
  //   P_req = clamp( P_loss(T_sub) + Kp * max(T_target - T_sub, 0), 0, P_max )

  if (!job_active_ || T_target_K_ <= 310.0) {
    P_requested_W_ = 0.0;
    return P_requested_W_;
  }

  const double err_K = (T_target_K_ - T_sub_K_);

  // Feed-forward term offsets current losses at current temperature
  const double P_ff = lossPowerW(T_sub_K_);

  // Proportional term requests additional power only when below target
  const double P_p = (err_K > 0.0) ? (Kp_W_per_K_ * err_K) : 0.0;

  double req = P_ff + P_p;
  if (!std::isfinite(req)) req = 0.0;

  req = std::clamp(req, 0.0, maxPower_W_);

  P_requested_W_ = req;
  return P_requested_W_;
}

void SubstrateHeater::applyHeat(double watts, double dt_s) {
  // Thermal integrator:
  //   C * dT/dt = P_in - P_loss(T)
  //   T_{k+1} = T_k + ( (P_in - P_loss(T_k)) / C ) * dt

  const double dt = std::max(0.0, dt_s);

  double Pin = std::max(0.0, watts);
  if (!std::isfinite(Pin)) Pin = 0.0;

  P_delivered_W_ = Pin;

  const double Ploss = lossPowerW(T_sub_K_);
  const double net_W = Pin - Ploss;

  const double dT = (net_W / C_J_per_K_) * dt;
  T_sub_K_ += dT;

  if (!std::isfinite(T_sub_K_)) T_sub_K_ = T_env_K_;
  if (T_sub_K_ < 0.0)           T_sub_K_ = 0.0;
}

bool SubstrateHeater::isAtTarget() const {
  // When inactive, considered ready to avoid blocking schedules.
  if (!job_active_ || T_target_K_ <= 310.0) return true;
  return (T_sub_K_ >= (T_target_K_ - READY_BAND_K_));
}

void SubstrateHeater::tick(const TickContext& ctx) {
  // Miss streak updates are evaluated after applyHeat(...) has been called for this tick.

  if (job_active_ && T_target_K_ > 310.0) {
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

  // Leader-only logging to avoid MPI multi-writer corruption.
  if (is_leader_) {
    Logger::instance().log_wide(
      name_,
      ctx.tick_index,
      ctx.time,
      {
        "job_index",
        "job_active",
        "T_sub_K",
        "T_target_K",
        "P_requested_W",
        "P_delivered_W",
        "temp_miss_streak",
        "job_failed"
      },
      {
        static_cast<double>(job_index_),
        job_active_ ? 1.0 : 0.0,
        T_sub_K_,
        T_target_K_,
        P_requested_W_,
        P_delivered_W_,
        static_cast<double>(temp_miss_streak_),
        job_failed_ ? 1.0 : 0.0
      }
    );
  }
}
