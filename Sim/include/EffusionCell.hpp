// Sim/include/EffusionCell.hpp
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

    // Target crucible temperature (K) implied by the job's flux.
    void setTargetTempK(double T_K) { target_temp_K_ = T_K; }

    // Read-only accessors used by main.cpp / diagnostics
    double getTemperature() const { return temperature_; }
    double getTargetTempK() const { return target_temp_K_; }

    // Actual heater power that was applied on the last tick (W).
    double getLastHeatInputW() const { return last_heat_W_; }

private:
    double temperature_{300.0};     // K (achieved crucible temperature)
    double target_temp_K_{300.0};   // K (desired temperature from flux schedule)

    double last_heat_W_{0.0};       // last-applied power (from HeaterBank)
    double heat_input_w_{0.0};      // what we log as heatInput

    // Energy balance tracking for diagnostics
    double last_p_loss_W_{0.0};     // W (Heat lost to space this tick)
    double last_net_W_{0.0};        // W (Net power available for dT)

    // Physical constants synced with main.cpp shadow proxy
    double c_j_per_k_ = 800.0;      // Thermal mass
    double h_w_per_k_ = 0.8;        // Insulation loss factor

    // Push-to-SPARTA cadence
    int    push_every_ticks_{10};
    double push_threshold_K_{1.0};
    double last_pushed_temp_{300.0};

    WakeChamber* sparta_ctrl_{nullptr};

    // SPARTA diagnostic CSV path
    std::filesystem::path diag_path_ =
        std::filesystem::path("data") / "tmp" / "effusion_diag.csv";
};