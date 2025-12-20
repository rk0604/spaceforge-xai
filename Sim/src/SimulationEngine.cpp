#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "PowerBus.hpp"
#include "SolarArray.hpp"
#include "HeaterBank.hpp"
#include "Logger.hpp"

void SimulationEngine::addSubsystem(Subsystem* subsystem) {
    subsystems_.push_back(subsystem);
}

void SimulationEngine::initialize() {
    // Identify “well-known” subsystems and initialize all
    for (auto* s : subsystems_) {
        if (auto* b  = dynamic_cast<Battery*>(s))    battery_  = b;
        if (auto* sa = dynamic_cast<SolarArray*>(s)) solar_   = sa;
        if (auto* pb = dynamic_cast<PowerBus*>(s))   powerbus_ = pb;
        if (auto* hb = dynamic_cast<HeaterBank*>(s)) heater_   = hb;
    }

    for (auto* s : subsystems_) s->initialize();

    job_failed_flag_ = false;

    logRow_(/*tick*/0, /*time*/0.0);

    tick_count_ = 1;
    sim_time_   = tick_step_;
}

void SimulationEngine::tick() {
    const TickContext ctx{ tick_count_, sim_time_, tick_step_ };

    for (auto* s : subsystems_) s->tick(ctx);

    logRow_(tick_count_, sim_time_);
    job_failed_flag_ = false;

    tick_count_ += 1;
    sim_time_   += tick_step_;
}

void SimulationEngine::setTickStep(double dt) {
    tick_step_ = dt;
}

void SimulationEngine::markJobFailedThisTick() {
    job_failed_flag_ = true;
}

void SimulationEngine::shutdown() {
    for (auto* s : subsystems_) s->shutdown();
}

void SimulationEngine::logRow_(int tick, double time) {
    double bus   = 0.0;
    double batt  = 0.0;
    double solar = 0.0;

    // Regime metadata (logged every row)
    double batt_cap_Wh = 0.0;
    double batt_cW     = 0.0;
    double batt_dW     = 0.0;

    double sol_eff     = 0.0;
    double sol_base_W  = 0.0;

    if (powerbus_) bus = powerbus_->getAvailablePower();

    if (battery_) {
        batt        = battery_->getCharge();
        // These getters will be added in Battery.hpp next.
        batt_cap_Wh = battery_->getCapacityWh();
        batt_cW     = battery_->getMaxChargeW();
        batt_dW     = battery_->getMaxDischargeW();
    }

    if (solar_) {
        solar      = solar_->getLastOutput();
        sol_eff    = solar_->getEfficiency();
        sol_base_W = solar_->getBaseInputW();
    }

    const double job_failed = job_failed_flag_ ? 1.0 : 0.0;

    Logger::instance().log_wide(
        "SimulationEngine",
        tick,
        time,
        {
            "status","bus","battery","solar","job_failed",
            "battery_capacity_Wh","battery_max_charge_W","battery_max_discharge_W",
            "solar_efficiency","solar_base_input_W"
        },
        {
            1.0, bus, batt, solar, job_failed,
            batt_cap_Wh, batt_cW, batt_dW,
            sol_eff, sol_base_W
        }
    );
}
