#pragma once
#include "Subsystem.hpp"
#include "TickContext.hpp"

class PowerBus : public Subsystem {
public:
    PowerBus();

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void addPower(double watts);

    // request power from bus (logs with context)
    double drawPower(double requested, const TickContext& ctx);

    double getAvailablePower() const;

private:
    double available_power_{0.0};
};
