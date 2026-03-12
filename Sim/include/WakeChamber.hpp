#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <mpi.h>

#include "TickContext.hpp"

// Forward declaration to keep the header lightweight.
class SpartaBridge;

/*
    WakeChamber is the C++ wrapper around the SPARTA wake simulation.

    Main responsibilities:
    - Own a SpartaBridge instance
    - Load and reload the SPARTA deck
    - Advance SPARTA by fixed blocks of steps
    - Rewrite params.inc when wake parameters change
    - Expose a per-tick logging hook for wake diagnostics

    Important MPI note:
    In multi-rank runs, all ranks must issue the same SPARTA command sequence.
    For that reason, the collective reload-and-advance method should be used
    instead of the old local-only reload method.
*/
class WakeChamber {
public:
    explicit WakeChamber(MPI_Comm comm, std::string label);
    ~WakeChamber();

    /*
        Initialize the wake chamber and load the SPARTA deck.

        deck_basename:
            SPARTA deck filename, for example "in.wake_harness"

        input_subdir:
            Input directory used by the deck and related SPARTA files
    */
    void init(const std::string& deck_basename, const std::string& input_subdir);

    /*
        Advance one outer wake tick.

        The first positive nDefault seen is remembered as the SPARTA block size.
        After that, each call advances by the remembered block size.
    */
    void step(int nDefault);

    /*
        Advance SPARTA by exactly n steps.
    */
    void runSteps(int n);

    /*
        Mark the wake chamber as needing a SPARTA deck reload before the next advance.

        This only sets a local flag. The actual reload decision should be made
        collectively across all MPI ranks before issuing SPARTA commands.
    */
    void markDirtyReload();

    /*
        Legacy local reload-or-advance method.

        This works in single-rank runs, but it is not MPI-safe if different ranks
        disagree about whether a reload is needed.
    */
    bool runIfDirtyOrAdvance(int n);

    /*
        MPI-safe reload-or-advance method.

        If any rank reports that a reload is needed, all ranks reload together.
        Otherwise, all ranks simply advance together.

        This is the method that should be used in multi-rank SPARTA runs.
    */
    bool runIfDirtyOrAdvanceCollective(int n);

    /*
        Shut down the wake chamber and release the SpartaBridge.
    */
    void shutdown();

    /*
        Update one wake parameter and rewrite params.inc.

        Only rank 0 writes the file.

        Important MPI note:
        This method does not perform a barrier or any other collective operation.
        Synchronization is handled later by runIfDirtyOrAdvanceCollective(), where
        all ranks participate before the SPARTA deck reload.
    */
    void setParameter(const std::string& name, double value);

    /*
        Per-engine-tick logging hook.

        This reads SPARTA diagnostics and writes a wide logging row for the current tick.
    */
    void tick(const TickContext& ctx);

private:
    /*
        Log wake lifecycle and reload events.

        Only rank 0 writes these event rows.
    */
    void logEvent_(double status,
                   double ran_steps,
                   double cum_steps,
                   double reload,
                   double mark_reload);

private:
    MPI_Comm comm_;
    std::string label_;

    std::unique_ptr<SpartaBridge> sp_{};

    bool initialized_{false};
    bool dirtyReload_{false};

    int cum_steps_{0};
    int last_run_steps_{0};
    long long event_id_{0};

    std::string deck_;
    std::string input_subdir_;

    // SPARTA diagnostic CSV written by the deck every block of steps.
    std::filesystem::path diag_path_ =
        std::filesystem::path("data") / "tmp" / "wake_diag.csv";
};