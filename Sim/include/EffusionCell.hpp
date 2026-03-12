#pragma once

#include <filesystem>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class WakeChamber; // forward declaration

class EffusionCell : public Subsystem {
public:
    /*
        Thermal-band classification for scheduler/control logic.

        Why this exists:
        The old interface only exposed a one-sided readiness check
        ("hot enough"), which was sufficient for warmup gating but not for the
        new scheduler semantics. The scheduler now needs to distinguish:

        - BelowTargetBand  -> warmup still needed
        - WithinTargetBand -> thermally ready for live deposition
        - AboveTargetBand  -> cooldown still needed
        - Idle             -> no meaningful deposition/heating target exists

        main.cpp should use this to decide whether the controlling job is in
        warmup, cooldown, generic thermal prep, or live deposition.
    */
    enum class ThermalBandState {
        Idle = 0,
        BelowTargetBand,
        WithinTargetBand,
        AboveTargetBand
    };

    EffusionCell() : Subsystem("EffusionCell") {}

    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

    // Called by HeaterBank to apply heater power in watts for dt seconds.
    void applyHeat(double watts, double dt);

    // Optional hookup to push parameters into the wake SPARTA instance.
    void setSpartaCtrl(WakeChamber* wc) { sparta_ctrl_ = wc; }

    // Set the desired effusion-cell target temperature in kelvin.
    // main.cpp uses this to express the process target for the current tick.
    void setTargetTempK(double T_K);

    // Read-only accessors used by main.cpp and diagnostics.
    double getTemperatureK() const { return temperature_; }
    double getTargetTempK() const { return target_temp_K_; }

    // Backward-compatible accessor in case existing code still uses this name.
    double getTemperature() const { return getTemperatureK(); }

    // Actual heater power applied during the most recent applyHeat() call.
    double getLastHeatInputW() const { return last_heat_W_; }

    /*
        Returns true when the current target is a real heating target rather
        than the idle baseline.

        This prevents 300 K idle states from being treated as
        deposition-readiness targets.
    */
    bool hasMeaningfulTarget() const;

    /*
        Backward-compatible one-sided readiness check.

        Existing behavior is preserved:
        - If the target is idle or not meaningful, returns true.
        - Otherwise requires temperature >= readiness_fraction * target.

        Important:
        This function does NOT distinguish "too hot" from "ready".
        Scheduler code that needs warmup vs cooldown semantics should use
        getThermalBandState() instead.
    */
    bool isAtTarget(double readiness_fraction = 0.90) const;

    /*
        Scheduler-grade thermal-band query.

        lower_readiness_fraction:
          Minimum acceptable fraction of target temperature to be considered in-band.

        upper_readiness_fraction:
          Maximum acceptable fraction of target temperature to be considered in-band.
          Values above this are treated as overheated for the current target and
          therefore imply cooldown rather than readiness.

        Default semantics:
        - Idle target       -> Idle
        - T < lower*target  -> BelowTargetBand
        - lower*target <= T <= upper*target -> WithinTargetBand
        - T > upper*target  -> AboveTargetBand

        Notes:
        - Fractions are clamped to sensible ranges internally.
        - If upper_readiness_fraction < lower_readiness_fraction, it will be
          promoted to the lower value to avoid invalid bands.
    */
    ThermalBandState getThermalBandState(double lower_readiness_fraction = 0.90,
                                         double upper_readiness_fraction = 1.05) const;

    // Convenience helpers for clearer scheduler code in main.cpp.
    bool isBelowTargetBand(double lower_readiness_fraction = 0.90) const;
    bool isWithinTargetBand(double lower_readiness_fraction = 0.90,
                            double upper_readiness_fraction = 1.05) const;
    bool isAboveTargetBand(double upper_readiness_fraction = 1.05) const;

private:
    // Actual effusion-cell temperature in kelvin.
    double temperature_{300.0};

    // Desired target temperature in kelvin.
    // This is provided by main.cpp from process control logic.
    double target_temp_K_{300.0};

    // Last-applied heater power in watts from HeaterBank.
    double last_heat_W_{0.0};

    // Per-tick heat input reported into logging.
    double heat_input_w_{0.0};

    // Thermal diagnostics logged each tick.
    double last_p_loss_W_{0.0};
    double last_net_W_{0.0};

    // Thermal model constants.
    double c_j_per_k_{800.0};
    double h_w_per_k_{0.8};

    // Push-to-SPARTA cadence controls.
    int    push_every_ticks_{10};
    double push_threshold_K_{1.0};
    double last_pushed_temp_{300.0};

    WakeChamber* sparta_ctrl_{nullptr};

    // Optional diagnostic path retained for compatibility with the existing class.
    std::filesystem::path diag_path_ =
        std::filesystem::path("data") / "tmp" / "effusion_diag.csv";
};