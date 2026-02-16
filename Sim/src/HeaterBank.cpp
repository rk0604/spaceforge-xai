#include "HeaterBank.hpp"
#include "EffusionCell.hpp"
#include "SubstrateHeater.hpp"
#include <algorithm>

HeaterBank::HeaterBank(double maxDraw)
: Subsystem("HeaterBank"), maxDraw_(maxDraw) {}

void HeaterBank::setPowerBus(PowerBus* bus) { bus_ = bus; }

void HeaterBank::setEffusionCell(EffusionCell* eff) { effusion_ = eff; }
void HeaterBank::setSubstrateHeater(SubstrateHeater* sub) { substrate_ = sub; }

void HeaterBank::setEffusionDemand(double watts) {
    std::lock_guard<std::mutex> lock(demandMtx_);
    effusionDemand_ = std::max(0.0, watts);
}

void HeaterBank::setSubstrateDemand(double watts) {
    std::lock_guard<std::mutex> lock(demandMtx_);
    substrateDemand_ = std::max(0.0, watts);
}

void HeaterBank::setPrioritySubstrate(bool priority) {
    prioritySubstrate_ = priority;
}

void HeaterBank::initialize() {}

void HeaterBank::tick(const TickContext& ctx) {
    if (!bus_) return;

    double effReq, subReq;

    {
        std::lock_guard<std::mutex> lock(demandMtx_);
        effReq = effusionDemand_;
        subReq = substrateDemand_;
    }

    // Respect bank max draw
    double totalRequested = effReq + subReq;
    if (totalRequested > maxDraw_) {
        double scale = maxDraw_ / totalRequested;
        effReq *= scale;
        subReq *= scale;
    }

    double effDelivered = 0.0;
    double subDelivered = 0.0;

    // ---- Priority allocation ----
    if (prioritySubstrate_) {
        subDelivered = bus_->drawPower(subReq, ctx);
        effDelivered = bus_->drawPower(effReq, ctx);
    } else {
        effDelivered = bus_->drawPower(effReq, ctx);
        subDelivered = bus_->drawPower(subReq, ctx);
    }

    // Apply heat
    if (effusion_) {
        effusion_->applyHeat(effDelivered, ctx.dt);
    }

    if (substrate_) {
        substrate_->applyHeat(subDelivered, ctx.dt);
    }

    Logger::instance().log_wide(
        name_,
        ctx.tick_index,
        ctx.time,
        {
            "eff_requested_W",
            "eff_delivered_W",
            "sub_requested_W",
            "sub_delivered_W",
            "priority_substrate"
        },
        {
            effReq,
            effDelivered,
            subReq,
            subDelivered,
            prioritySubstrate_ ? 1.0 : 0.0
        }
    );
}

void HeaterBank::shutdown() {}
