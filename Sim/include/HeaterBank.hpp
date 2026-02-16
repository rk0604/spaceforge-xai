#pragma once
#include "Subsystem.hpp"
#include "PowerBus.hpp"
#include "TickContext.hpp"
#include "Logger.hpp"
#include <mutex>

class EffusionCell;
class SubstrateHeater;

class HeaterBank : public Subsystem {
public:
    explicit HeaterBank(double maxDraw = 4000.0); // upgraded realistic max

    void setPowerBus(PowerBus* bus);

    void setEffusionCell(EffusionCell* eff);
    void setSubstrateHeater(SubstrateHeater* sub);

    void setEffusionDemand(double watts);
    void setSubstrateDemand(double watts);

    void setPrioritySubstrate(bool priority);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

private:
    PowerBus* bus_ = nullptr;
    EffusionCell* effusion_ = nullptr;
    SubstrateHeater* substrate_ = nullptr;

    double maxDraw_;

    double effusionDemand_  = 0.0;
    double substrateDemand_ = 0.0;

    bool prioritySubstrate_ = false;

    std::mutex demandMtx_;
};
