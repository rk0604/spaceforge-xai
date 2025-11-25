#pragma once

#include "SimulationEngine.hpp"   // for TickContext
#include "Subsystem.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class PowerBus;  // forward declaration

class GrowthMonitor : public Subsystem {
public:
    explicit GrowthMonitor(int gridN = 32);

    // Only rank 0 should actually accumulate and write CSV.
    void setIsLeader(bool v) { isLeader_ = v; }

    // Hook into the shared power bus so this instrument can draw power.
    void setPowerBus(PowerBus* bus) { bus_ = bus; }

    // Call once after jobs[] has been loaded.
    void setNumJobs(std::size_t nJobs);

    // Called once per tick (on leader) before engine.tick().
    // jobIndex < 0 means "no active job / background".
    void setBeamState(int jobIndex, bool mbeOn, double Fwafer_cm2s);

    // Optional: mark that a job was aborted early (still outputs wafer).
    void markJobAborted(int jobIndex);

    // Subsystem interface
    void initialize() override;
    void tick(const TickContext& ctx) override;
    void shutdown() override;

private:
    struct JobAccum {
        bool aborted        = false;
        bool had_growth     = false;
        double last_t_end_s = 0.0;
        std::vector<double> dose;  // length = gridN_ * gridN_
    };

    int    gridN_;
    double waferRadiusCells_;
    bool   isLeader_ = true;

    // Precomputed mask: 1 if cell center lies inside wafer circle, else 0.
    std::vector<std::uint8_t> waferMask_;

    // Per-tick beam state (pushed from main)
    int    activeJobIndex_    = -1;
    bool   mbeOn_             = false;
    double currentFwaferCm2s_ = 0.0;

    std::vector<JobAccum> jobs_;
    std::string csvFilename_;

    // Power coupling
    PowerBus* bus_           = nullptr;
    double    monitorPowerW_ = 5.0;  // fixed instrument load in watts

    void buildWaferMask();
    void ensureJobStorage();
    void integrateDose(double dt, double t_now);
    void writeCsv() const;
};
