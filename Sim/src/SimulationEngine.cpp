#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "PowerBus.hpp"
#include "SolarArray.hpp"
#include "HeaterBank.hpp"
#include "Logger.hpp"

void SimulationEngine::addSubsystem(Subsystem* subsystem) {
    subsystems_.push_back(subsystem);
}

void SimulationEngine::setBaseLoadW(double w) {
    base_load_W_ = w;
}

void SimulationEngine::initialize() {
    // Discover pointers we want to phase-order
    for (auto* s : subsystems_) {
        if (auto* b  = dynamic_cast<Battery*>(s))    battery_  = b;
        if (auto* sa = dynamic_cast<SolarArray*>(s)) solar_    = sa;
        if (auto* pb = dynamic_cast<PowerBus*>(s))   powerbus_ = pb;
        if (auto* hb = dynamic_cast<HeaterBank*>(s)) heater_   = hb;
    }

    for (auto* s : subsystems_) s->initialize();

    job_failed_flag_ = false;
    base_drawn_W_ = 0.0;

    latched_bus_remaining_W_   = 0.0;
    latched_total_requested_W_ = 0.0;
    latched_total_granted_W_   = 0.0;
    latched_total_generated_W_ = 0.0;

    logRow_(0, 0.0);

    tick_count_ = 1;
    sim_time_   = tick_step_;
}

void SimulationEngine::tick() {
    const TickContext ctx{ tick_count_, sim_time_, tick_step_ };

    // ------------------------------------------------------
    // Phase 0) Generation FIRST (solar adds into the bus)
    // ------------------------------------------------------
    if (solar_) {
        solar_->tick(ctx);
    }

    // ------------------------------------------------------
    // Phase 1) Spacecraft baseline draw AFTER generation exists
    // ------------------------------------------------------
    if (powerbus_ && base_load_W_ > 0.0) {
        base_drawn_W_ = powerbus_->drawPower(base_load_W_, ctx);
    } else {
        base_drawn_W_ = 0.0;
    }

    // ------------------------------------------------------
    // Phase 2) Tick everything EXCEPT Solar, PowerBus, Battery
    //   - Preserve the addSubsystem() order for the rest
    //   - HeaterBank should run before Effusion/Substrate ticks
    // ------------------------------------------------------
    for (auto* s : subsystems_) {
        if (s == solar_)     continue;
        if (s == powerbus_)  continue;
        if (s == battery_)   continue;
        s->tick(ctx);
    }

    // ------------------------------------------------------
    // Phase 3) LATCH PowerBus counters BEFORE PowerBus::tick()
    //   because PowerBus::tick() logs then resets counters.
    // ------------------------------------------------------
    if (powerbus_) {
        latched_bus_remaining_W_   = powerbus_->getAvailablePower();
        latched_total_requested_W_ = powerbus_->getRequestedThisTickW();
        latched_total_granted_W_   = powerbus_->getGrantedThisTickW();
        latched_total_generated_W_ = powerbus_->getAddedThisTickW();
    } else {
        latched_bus_remaining_W_   = 0.0;
        latched_total_requested_W_ = 0.0;
        latched_total_granted_W_   = 0.0;
        latched_total_generated_W_ = 0.0;
    }

    // ------------------------------------------------------
    // Phase 4) Bus settlement (surplus -> battery) + PowerBus log/reset
    // ------------------------------------------------------
    if (powerbus_) {
        powerbus_->tick(ctx);
    }

    // ------------------------------------------------------
    // Phase 5) Battery tick LAST so Battery.csv reflects post-settlement state
    // ------------------------------------------------------
    if (battery_) {
        battery_->tick(ctx);
    }

    // ------------------------------------------------------
    // Phase 6) Log system snapshot (uses latched bus totals)
    // ------------------------------------------------------
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

    double bus_remaining = latched_bus_remaining_W_;
    double batt_charge   = 0.0;
    double solar_output  = 0.0;

    double batt_cap_Wh = 0.0;
    double batt_cW     = 0.0;
    double batt_dW     = 0.0;

    double sol_eff     = 0.0;
    double sol_base_W  = 0.0;

    const double total_requested_W = latched_total_requested_W_;
    const double total_granted_W   = latched_total_granted_W_;
    const double total_generated_W = latched_total_generated_W_;

    if (battery_) {
        batt_charge = battery_->getCharge();
        batt_cap_Wh = battery_->getCapacityWh();
        batt_cW     = battery_->getMaxChargeW();
        batt_dW     = battery_->getMaxDischargeW();
    }

    if (solar_) {
        solar_output = solar_->getLastOutput();
        sol_eff      = solar_->getEfficiency();
        sol_base_W   = solar_->getBaseInputW();
    }

    const double job_failed = job_failed_flag_ ? 1.0 : 0.0;

    Logger::instance().log_wide(
        "SimulationEngine",
        tick,
        time,
        {
            "status",
            "bus_remaining_W",
            "battery_charge_Wh",
            "solar_output_W",
            "job_failed",

            "spacecraft_base_load_W",
            "total_power_requested_W",
            "total_power_granted_W",
            "total_power_generated_W",

            "battery_capacity_Wh",
            "battery_max_charge_W",
            "battery_max_discharge_W",

            "solar_efficiency",
            "solar_base_input_W"
        },
        {
            1.0,
            bus_remaining,
            batt_charge,
            solar_output,
            job_failed,

            base_drawn_W_,
            total_requested_W,
            total_granted_W,
            total_generated_W,

            batt_cap_Wh,
            batt_cW,
            batt_dW,

            sol_eff,
            sol_base_W
        }
    );
}
