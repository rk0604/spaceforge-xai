#pragma once
#include <filesystem>
#include <mutex>
#include "Subsystem.hpp"
#include "TickContext.hpp"

class WakeChamber; // fwd

class EffusionCell : public Subsystem {
public:
    EffusionCell() : Subsystem("EffusionCell") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    // Called by HeaterBank to apply power (watts) for dt seconds
    void applyHeat(double watts, double dt);

    // Optional hookup to push parameters into the wake SPARTA instance
    void setSpartaCtrl(WakeChamber* wc) { sparta_ctrl_ = wc; }

    // Read-only accessors used by main.cpp / diagnostics
    double getTemperature() const { return temperature_; }

    // Actual heater power that was applied on the last tick (W).
    // This is what main.cpp uses to compare against the requested power.
    double getLastHeatInputW() const { return last_heat_W_; }

private:
    double temperature_{300.0};     // K
    double last_heat_W_{0.0};       // last-applied power (from HeaterBank)
    double heat_input_w_{0.0};      // what we log as heatInput

    // Push-to-SPARTA cadence
    int    push_every_ticks_{10};
    double push_threshold_K_{1.0};
    double last_pushed_temp_{300.0};

    WakeChamber* sparta_ctrl_{nullptr};

    // SPARTA diagnostic CSV path
    std::filesystem::path diag_path_ =
        std::filesystem::path("data") / "tmp" / "effusion_diag.csv";
};
