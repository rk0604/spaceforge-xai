#pragma once

#include <filesystem>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class WakeChamber;

// -----------------------------------------------------------------------------
// EffusionCell
//
// Lightweight orbit-aware source thermal model used by the scheduler and heater
// control logic.
//
// Design goals:
// 1. Preserve the current scheduler semantics and readiness logic.
// 2. Keep the source model simple and publication-defensible.
// 3. Add orbit-aware environmental forcing without introducing a large
//    spacecraft thermal model.
// 4. Expose effective ambient temperature, solar scale, and absorbed solar
//    heating so main.cpp can log the source thermal environment explicitly.
//
// Thermal model summary:
//
//   T_env_eff = T_night + (T_day - T_night) * solar_scale
//   P_solar_abs = alpha_abs * A_proj * G_solar * solar_scale
//
//   P_loss = h * (T_cell - T_env_eff)
//   P_net  = P_heater + P_solar_abs - P_loss
//
//   dT/dt = P_net / C
//
// Notes:
// - The loss model remains first-order linear RC style.
// - Negative P_loss is allowed when T_cell < T_env_eff so the environment can
//   passively warm the source.
// - The scheduler-facing thermal band interface is unchanged.
// -----------------------------------------------------------------------------
class EffusionCell : public Subsystem {
public:
    /*
        Thermal-band classification for scheduler and control logic.

        State meanings:
        - Idle: no meaningful source target exists
        - BelowTargetBand: warmup still needed
        - WithinTargetBand: thermally ready for execution
        - AboveTargetBand: cooldown still needed
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

    // Called by HeaterBank to apply source heater power in watts for dt seconds.
    void applyHeat(double watts, double dt);

    // Optional hookup to push parameters into the wake SPARTA instance.
    void setSpartaCtrl(WakeChamber* wc) { sparta_ctrl_ = wc; }

    // Set the desired source target temperature in kelvin.
    void setTargetTempK(double T_K);

    /*
        Update the orbit-aware thermal environment seen by the source.

        solar_scale is expected to be in the range [0, 1] where:
        - 0 means eclipse / night-side environment
        - 1 means fully sunlit environment

        This updates:
        - ambient_temp_K_
        - solar_scale_
        - solar_absorbed_power_W_
    */
    void setOrbitThermalEnvironment(double solar_scale);

    // Read-only accessors used by main.cpp and diagnostics.
    double getTemperatureK() const { return temperature_; }
    double getTargetTempK() const { return target_temp_K_; }

    // Backward-compatible accessor.
    double getTemperature() const { return getTemperatureK(); }

    // Actual heater power applied during the most recent applyHeat() call.
    double getLastHeatInputW() const { return last_heat_W_; }

    /*
        Effective ambient or environment temperature currently affecting the
        source due to the orbit-aware solar-scale model.

        In logging, this is the source-side effective ambient temperature seen
        by the source at the current tick.
    */
    double getAmbientTempK() const { return ambient_temp_K_; }

    // Current orbit-driven solar scale used by the source thermal model.
    double getSolarScale() const { return solar_scale_; }

    // Current absorbed solar heating power used by the source thermal model.
    double getSolarAbsorbedPowerW() const { return solar_absorbed_power_W_; }

    /*
        Returns true when the current target is a real process target rather
        than the idle baseline.
    */
    bool hasMeaningfulTarget() const;

    /*
        Backward-compatible one-sided readiness check.

        Behavior:
        - If the target is idle or not meaningful, returns true.
        - Otherwise requires temperature >= readiness_fraction * target.
    */
    bool isAtTarget(double readiness_fraction = 0.90) const;

    /*
        Scheduler-grade thermal-band query.

        lower_readiness_fraction:
          Minimum acceptable fraction of target temperature to be considered
          in-band.

        upper_readiness_fraction:
          Maximum acceptable fraction of target temperature to be considered
          in-band.

        Semantics:
        - Idle target -> Idle
        - T < lower * target -> BelowTargetBand
        - lower * target <= T <= upper * target -> WithinTargetBand
        - T > upper * target -> AboveTargetBand
    */
    ThermalBandState getThermalBandState(double lower_readiness_fraction = 0.90,
                                         double upper_readiness_fraction = 1.05) const;

    // Convenience helpers for clearer scheduler code in main.cpp.
    bool isBelowTargetBand(double lower_readiness_fraction = 0.90) const;
    bool isWithinTargetBand(double lower_readiness_fraction = 0.90,
                            double upper_readiness_fraction = 1.05) const;
    bool isAboveTargetBand(double upper_readiness_fraction = 1.05) const;

private:
    // Actual source temperature in kelvin.
    double temperature_{300.0};

    // Desired process target temperature in kelvin.
    double target_temp_K_{300.0};

    // Last-applied heater power in watts from HeaterBank.
    double last_heat_W_{0.0};

    // Per-tick heat input reported into logging.
    double heat_input_w_{0.0};

    // Thermal diagnostics logged each tick.
    double last_p_loss_W_{0.0};
    double last_net_W_{0.0};

    /*
        Orbit-aware environmental state.

        ambient_temp_K_:
          Effective environment temperature currently affecting the source.

        solar_scale_:
          Orbit-driven illumination factor used this tick.

        solar_absorbed_power_W_:
          Absorbed solar heating power currently acting on the source.
    */
    double ambient_temp_K_{300.0};
    double solar_scale_{0.0};
    double solar_absorbed_power_W_{0.0};

    // Core first-order source thermal model constants.
    double c_j_per_k_{800.0};
    double h_w_per_k_{0.8};

    /*
        Simple orbit-aware environmental model constants.

        These are intentionally lightweight and tunable. They represent an
        effective thermal environment seen by the source rather than a full
        spacecraft thermal model.
    */
    double night_ambient_temp_K_{285.0};
    double day_ambient_temp_K_{325.0};
    double solar_absorptivity_{0.35};
    double projected_area_m2_{0.010};
    double solar_constant_W_m2_{1361.0};

    // Push-to-SPARTA cadence controls.
    int    push_every_ticks_{10};
    double push_threshold_K_{1.0};
    double last_pushed_temp_{300.0};

    WakeChamber* sparta_ctrl_{nullptr};

    // Optional diagnostic path retained for compatibility.
    std::filesystem::path diag_path_ =
        std::filesystem::path("data") / "tmp" / "effusion_diag.csv";
};