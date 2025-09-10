#include "HeaterBank.hpp"
#include "EffusionCell.hpp"
#include "Logger.hpp"

HeaterBank::HeaterBank(double maxDraw)
: Subsystem("HeaterBank"), maxDraw_(maxDraw) {}

void HeaterBank::setPowerBus(PowerBus* bus) { bus_ = bus; }
void HeaterBank::setDemand(double watts) {
    std::lock_guard<std::mutex> lock(demandMtx_);
    demand_ = watts;
}
void HeaterBank::setEffusionCell(EffusionCell* eff) { effusion_ = eff; }

void HeaterBank::initialize() {
    // No file row here – keep logs per-tick only
}

void HeaterBank::tick(const TickContext& ctx) {
    if (!bus_) return;

    double requested;
    {
        std::lock_guard<std::mutex> lock(demandMtx_);
        requested = std::min(demand_, maxDraw_);
    }

    double granted = bus_->drawPower(requested, ctx);
    lastConsumed_ = granted;

    if (effusion_) {
        effusion_->applyHeat(granted, ctx.dt); // pass power (W)
    }

    // One wide row per tick with a fixed schema
    Logger::instance().log_wide(name_, ctx.tick_index, ctx.time,
        {"requested","granted"},
        {requested, granted});
}

void HeaterBank::shutdown() {
    // No file row here
}
