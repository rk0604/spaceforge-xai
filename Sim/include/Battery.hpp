#pragma once
#include "Subsystem.hpp"
#include "PowerBus.hpp"

class Battery : public Subsystem {
public:
    explicit Battery(double capacity = 1000.0);

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    void setPowerBus(PowerBus* bus);
    double getCharge() const;

private:
    PowerBus* bus_;
    double capacity_;    // maximum storage (Wh)
    double charge_;      // current charge (Wh)

    // policy: how much we try to draw each tick to charge up
    double request_per_tick_ = 5.0;  // matches your previous logs
};
