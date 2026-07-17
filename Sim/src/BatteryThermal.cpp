#include "BatteryThermal.hpp"

#include "Battery.hpp"
#include "Logger.hpp"
#include "PowerBus.hpp"
#include "Radiator.hpp"

#include <algorithm>
#include <cmath>

/*
    BatteryThermal

    See BatteryThermal.hpp for the model summary.

    Tick ordering contract

    This node ticks after the heater loads and before PowerBus bookkeeping,
    so its survival heater draw participates in this tick's power accounting.
    Joule heating uses the previous tick's battery flows (snapshotted inside
    Battery::tick), which is a deliberate one-tick lag.

    It also ticks before the Radiator so the pack's rejected heat lands on
    the coolant loop within the same tick.
*/

namespace {
double clampFinite(double v, double lo, double hi, double fallback) {
    if (!std::isfinite(v)) return fallback;
    return std::clamp(v, lo, hi);
}
} // namespace

void BatteryThermal::initialize() {
    pack_temp_K_ = 293.0;
    survival_on_ = false;

    Logger::instance().log_wide(
        "BatteryThermal", 0, 0.0,
        {
            "status",
            "T_batt_K",
            "T_sink_K",
            "q_joule_W",
            "q_cool_W",
            "batt_current_A",
            "surv_heater_req_W",
            "surv_heater_granted_W",
            "derate_discharge",
            "derate_charge",
            "prev_discharge_W",
            "prev_charge_W"
        },
        {1.0, pack_temp_K_, fallback_sink_K_, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0}
    );
}

void BatteryThermal::tick(const TickContext& ctx) {
    // ---- 1) Joule self-heating from the previous tick's battery flows ----
    double prev_dis_W = 0.0;
    double prev_chg_W = 0.0;

    if (battery_) {
        prev_dis_W = battery_->getPrevTickDischargeW();
        prev_chg_W = battery_->getPrevTickChargeW();
    }

    const double i_batt_A = (prev_dis_W + prev_chg_W) / v_bus_nom_V_;
    const double q_joule_W = i_batt_A * i_batt_A * r_internal_ohm_;

    // ---- 2) Conductive cooling into the radiator coolant loop ----
    const double sink_K =
        radiator_ ? radiator_->getLoopTempK() : fallback_sink_K_;

    const double q_cool_W = h_pack_W_per_K_ * (pack_temp_K_ - sink_K);

    // Heat leaving the pack lands on the radiator loop.
    if (radiator_ && q_cool_W > 0.0) {
        radiator_->addHeatLoad(q_cool_W);
    }

    // ---- 3) Survival heater (bang-bang with hysteresis) ----
    if (pack_temp_K_ < survival_on_K_) {
        survival_on_ = true;
    } else if (pack_temp_K_ >= survival_off_K_) {
        survival_on_ = false;
    }

    const double surv_req_W = survival_on_ ? survival_power_W_ : 0.0;

    double surv_granted_W = 0.0;
    if (bus_ && surv_req_W > 0.0) {
        surv_granted_W = bus_->drawPower(surv_req_W, ctx);
    }

    // ---- 4) Integrate pack temperature (explicit Euler) ----
    const double net_W = q_joule_W + surv_granted_W - q_cool_W;
    pack_temp_K_ += (net_W * ctx.dt) / c_pack_J_per_K_;
    pack_temp_K_ = clampFinite(pack_temp_K_, 200.0, 400.0, 293.0);

    // ---- 5) Push thermal derating into the electrical battery model ----
    double derate_dis = 0.5 + 0.5 * (pack_temp_K_ - t_dis_lo_K_) /
                                   (t_dis_hi_K_ - t_dis_lo_K_);
    derate_dis = clampFinite(derate_dis, 0.5, 1.0, 1.0);

    double derate_chg = 1.0 - 0.6 * (pack_temp_K_ - t_chg_lo_K_) /
                                   (t_chg_hi_K_ - t_chg_lo_K_);
    derate_chg = clampFinite(derate_chg, 0.4, 1.0, 1.0);

    if (battery_) {
        battery_->setThermalDerating(derate_dis, derate_chg);
    }

    Logger::instance().log_wide(
        "BatteryThermal", ctx.tick_index, ctx.time,
        {
            "status",
            "T_batt_K",
            "T_sink_K",
            "q_joule_W",
            "q_cool_W",
            "batt_current_A",
            "surv_heater_req_W",
            "surv_heater_granted_W",
            "derate_discharge",
            "derate_charge",
            "prev_discharge_W",
            "prev_charge_W"
        },
        {
            1.0,
            pack_temp_K_,
            sink_K,
            q_joule_W,
            q_cool_W,
            i_batt_A,
            surv_req_W,
            surv_granted_W,
            derate_dis,
            derate_chg,
            prev_dis_W,
            prev_chg_W
        }
    );
}

void BatteryThermal::shutdown() {}
