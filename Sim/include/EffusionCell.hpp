#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "Subsystem.hpp"
#include "TickContext.hpp"

class WakeChamber;

/*
    EffusionCell

    Lightweight orbit-aware source thermal model used by the scheduler and
    heater control logic.

    Design goals

    1. Preserve scheduler semantics and readiness logic.
    2. Keep the source model simple and publication defensible.
    3. Support runtime configuration from the analytics tracker.
    4. Expose effective ambient temperature, solar scale, and absorbed solar
       heating so main.cpp can log the source thermal environment explicitly.

    Thermal model summary

        T_env_eff = T_night + (T_day - T_night) * solar_scale

        P_solar_abs = alpha_abs * A_proj * G_solar * solar_scale

        P_loss = h * (T_cell - T_env_eff)

        P_net = P_heater + P_solar_abs - P_loss

        dT / dt = P_net / C

    Notes

    The loss model remains first order and linear.

    Negative P_loss is allowed when T_cell is below T_env_eff, which means the
    environment can passively warm the source.

    Runtime constants such as C_J, h_WK, and day or night ambient temperature
    should be passed through command line arguments rather than edited inside
    this header for each config.
*/
class EffusionCell : public Subsystem {
public:
    /*
        Thermal-band classification for scheduler and control logic.

        State meanings

        Idle
            No meaningful source target exists.

        BelowTargetBand
            Warmup is still needed.

        WithinTargetBand
            Source is thermally ready for execution.

        AboveTargetBand
            Cooldown is still needed.
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
        Fractional target-temperature bias pushed by SourceInventory.

        As the crucible charge depletes, the temperature required to sustain
        a given flux rises. The bias is applied inside setTargetTempK to
        every meaningful (above idle) scheduler target:

            T_target_eff = T_target * (1 + bias)

        Clamped to [0, 0.5]. Defaults to 0 so the cell behaves exactly as
        before when no SourceInventory node is wired.
    */
    void setInventoryTempBiasFrac(double bias_frac) {
        if (std::isfinite(bias_frac)) {
            inventory_temp_bias_frac_ = std::clamp(bias_frac, 0.0, 0.5);
        }
    }

    double getInventoryTempBiasFrac() const { return inventory_temp_bias_frac_; }

    /*
        Configure the core source thermal constants.

        c_j_per_k
            Lumped thermal capacitance in joules per kelvin.

        h_w_per_k
            Lumped linear heat loss coefficient in watts per kelvin.

        This method is intentionally lightweight. Invalid values are ignored so
        a malformed command line argument cannot silently destroy the thermal
        model state.
    */
    void setThermalConstants(double c_j_per_k, double h_w_per_k) {
        if (std::isfinite(c_j_per_k) && c_j_per_k > 0.0) {
            c_j_per_k_ = c_j_per_k;
        }

        if (std::isfinite(h_w_per_k) && h_w_per_k > 0.0) {
            h_w_per_k_ = h_w_per_k;
        }
    }

    /*
        Configure the orbit-aware effective ambient temperature limits.

        night_ambient_temp_K
            Effective source environment during eclipse.

        day_ambient_temp_K
            Effective source environment during full sunlight.

        After updating the constants, the current orbit thermal environment is
        recomputed using the already stored solar scale.
    */
    void setEnvironmentConstants(double night_ambient_temp_K,
                                 double day_ambient_temp_K) {
        if (std::isfinite(night_ambient_temp_K) && night_ambient_temp_K > 0.0) {
            night_ambient_temp_K_ = night_ambient_temp_K;
        }

        if (std::isfinite(day_ambient_temp_K) && day_ambient_temp_K > 0.0) {
            day_ambient_temp_K_ = day_ambient_temp_K;
        }

        setOrbitThermalEnvironment(solar_scale_);
    }

    /*
        Update the orbit-aware thermal environment seen by the source.

        solar_scale is expected to be in the range from 0 to 1.

        0 means eclipse or night-side environment.

        1 means fully sunlit environment.

        This updates three values:

        ambient_temp_K_
            Effective ambient temperature currently seen by the source.

        solar_scale_
            Clamped orbit illumination factor.

        solar_absorbed_power_W_
            Absorbed solar heating power currently added to the source node.
    */
    void setOrbitThermalEnvironment(double solar_scale);

    // Read-only accessors used by main.cpp and diagnostics.
    double getTemperatureK() const { return temperature_; }
    double getTargetTempK() const { return target_temp_K_; }

    // Backward-compatible accessor.
    double getTemperature() const { return getTemperatureK(); }

    // Actual heater power applied during the most recent applyHeat call.
    double getLastHeatInputW() const { return last_heat_W_; }

    /*
        Effective ambient or environment temperature currently affecting the
        source due to the orbit-aware solar-scale model.
    */
    double getAmbientTempK() const { return ambient_temp_K_; }

    // Current orbit-driven solar scale used by the source thermal model.
    double getSolarScale() const { return solar_scale_; }

    // Current absorbed solar heating power used by the source thermal model.
    double getSolarAbsorbedPowerW() const { return solar_absorbed_power_W_; }

    // Runtime-configured source thermal capacitance.
    double getThermalCapacitanceJPerK() const { return c_j_per_k_; }

    // Runtime-configured linear heat loss coefficient.
    double getHeatLossWPerK() const { return h_w_per_k_; }

    // Runtime-configured eclipse effective ambient temperature.
    double getNightAmbientTempK() const { return night_ambient_temp_K_; }

    // Runtime-configured sunlit effective ambient temperature.
    double getDayAmbientTempK() const { return day_ambient_temp_K_; }

    /*
        Returns true when the current target is a real process target rather
        than the idle baseline.
    */
    bool hasMeaningfulTarget() const;

    /*
        Backward-compatible one-sided readiness check.

        Behavior

        If the target is idle or not meaningful, this returns true.

        Otherwise the source must reach at least readiness_fraction times the
        target temperature.
    */
    bool isAtTarget(double readiness_fraction = 0.90) const;

    /*
        Scheduler-grade thermal-band query.

        lower_readiness_fraction
            Minimum acceptable fraction of target temperature needed to be
            considered in-band.

        upper_readiness_fraction
            Maximum acceptable fraction of target temperature allowed before
            the source is considered too hot.

        Semantics

        Idle target
            Idle

        T < lower * target
            BelowTargetBand

        lower * target <= T <= upper * target
            WithinTargetBand

        T > upper * target
            AboveTargetBand
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

    // Fractional target bias from SourceInventory depletion (0 = fresh charge).
    double inventory_temp_bias_frac_{0.0};

    // Last-applied heater power in watts from HeaterBank.
    double last_heat_W_{0.0};

    // Per-tick heat input reported into logging.
    double heat_input_w_{0.0};

    // Thermal diagnostics logged each tick.
    double last_p_loss_W_{0.0};
    double last_net_W_{0.0};

    /*
        Orbit-aware environmental state.

        ambient_temp_K_
            Effective environment temperature currently affecting the source.

        solar_scale_
            Orbit-driven illumination factor used this tick.

        solar_absorbed_power_W_
            Absorbed solar heating power currently acting on the source.
    */
    double ambient_temp_K_{300.0};
    double solar_scale_{0.0};
    double solar_absorbed_power_W_{0.0};

    /*
        Core first-order source thermal model constants.

        Config 1 fallback values:
            C_J = 800 J per K
            h_WK = 0.8 W per K
    */
    double c_j_per_k_{800.0};
    double h_w_per_k_{0.8};

    /*
        Simple orbit-aware environmental model constants.

        These represent an effective thermal environment seen by the source
        rather than a full spacecraft thermal model.

        Corrected fallback values:
            eclipse ambient = 250 K
            sunlit ambient = 325 K
    */
    double night_ambient_temp_K_{250.0};
    double day_ambient_temp_K_{325.0};

    /*
        Solar absorption model constants.

        These are not currently part of the analytics tracker. They remain as
        stable simulator constants unless a later experiment requires them to be
        promoted to runtime parameters.
    */
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