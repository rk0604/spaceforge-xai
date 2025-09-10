#pragma once
#include <memory>
#include <string>
#include <filesystem>
#include <mpi.h>

#include "TickContext.hpp"   // ensures signature matches
// forward declare to keep header light
class SpartaBridge;

class WakeChamber {
public:
    explicit WakeChamber(MPI_Comm comm, std::string label);
    ~WakeChamber();

    void init(const std::string& deck_basename, const std::string& input_subdir);
    void step(int nDefault);
    void runSteps(int n);
    void markDirtyReload();
    bool runIfDirtyOrAdvance(int n);
    void shutdown();

    void setParameter(const std::string& name, double value);

    // per-engine-tick hook
    void tick(const TickContext& ctx);

private:
    void logEvent_(double status, double ran_steps, double cum_steps,
                   double reload, double mark_reload);

private:
    MPI_Comm comm_;
    std::string label_;

    std::unique_ptr<SpartaBridge> sp_{};

    bool initialized_{false};
    bool dirtyReload_{false};

    int  cum_steps_{0};
    int  last_run_steps_{0};
    long long event_id_{0};

    std::string deck_;
    std::string input_subdir_;

    // SPARTA diagnostic CSV that the deck writes every N steps
    std::filesystem::path diag_path_ = std::filesystem::path("data") / "tmp" / "wake_diag.csv";
};
