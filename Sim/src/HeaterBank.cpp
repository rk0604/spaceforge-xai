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

    double requested_w;
    {
        std::lock_guard<std::mutex> lock(demandMtx_);
        requested_w = std::min(demand_, maxDraw_);
    }

    double received_w = bus_->drawPower(requested_w, ctx);
    lastConsumed_ = received_w;

    if (effusion_) {
        // Pass the *actual* received power to the effusion cell
        effusion_->applyHeat(received_w, ctx.dt);
    }

    // One wide row per tick with a fixed schema
    // Columns renamed for clarity: requested_w, received_w (both in watts)
    Logger::instance().log_wide(
        name_,
        ctx.tick_index,
        ctx.time,
        {"requested_w", "received_w"},
        {requested_w,   received_w}
    );
}

void HeaterBank::shutdown() {
    // No file row here
}
