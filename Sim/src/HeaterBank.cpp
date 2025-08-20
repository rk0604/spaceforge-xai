#include "HeaterBank.hpp"
#include "EffusionCell.hpp" // only if exists, otherwise forward is fine

HeaterBank::HeaterBank(double maxDraw)
: Subsystem("HeaterBank"), maxDraw_(maxDraw) {}

void HeaterBank::setPowerBus(PowerBus* bus) { bus_ = bus; }
void HeaterBank::setDemand(double watts) {
    std::lock_guard<std::mutex> lock(demandMtx_);
    demand_ = watts;
}
void HeaterBank::setEffusionCell(EffusionCell* eff) { effusion_ = eff; }

void HeaterBank::initialize() {
    Logger::instance().log(name_, 0, 0.0, {{"status", 1}});
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

    // only apply heat if effusion cell exists
    if (effusion_) {
        double energyJoules = granted * ctx.dt;
        effusion_->applyHeat(energyJoules, ctx.dt);
    }

    Logger::instance().log(name_, ctx.tick_index, ctx.time, {
        {"requested", requested},
        {"granted", granted}
    });
}

void HeaterBank::shutdown() {
    Logger::instance().log(name_, -1, -1.0, {{"status", 0}});
}
