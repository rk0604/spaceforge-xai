#pragma once
#include "Subsystem.hpp"
#include "PowerBus.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"
#include <mutex>

class EffusionCell; // forward declare

class HeaterBank : public Subsystem {
public:
    HeaterBank(double maxDraw = 200.0);

    void setPowerBus(PowerBus* bus);
    void setDemand(double watts);
    void setEffusionCell(EffusionCell* eff);  // optional hookup

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

private:
    PowerBus* bus_ = nullptr;
    EffusionCell* effusion_ = nullptr;

    double maxDraw_;
    double demand_ = 0.0;
    double lastConsumed_ = 0.0;

    std::mutex demandMtx_;
};
