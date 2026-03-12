#include "WakeChamber.hpp"
#include "SpartaBridge.hpp"
#include "Logger.hpp"
#include "SpartaDiag.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <sstream>
#include <optional>
#include <limits>
#include <map>
#include <mpi.h>

namespace fs = std::filesystem;

/*
    This file manages the SPARTA-backed wake chamber subsystem.

    High level responsibilities:

    1. Own and control a SpartaBridge instance.
    2. Load the SPARTA deck once at startup.
    3. Advance the SPARTA simulation in fixed-size blocks.
    4. Rewrite params.inc when external code changes wake parameters.
    5. Reload the SPARTA deck when parameters change.
    6. Log wake diagnostics and lifecycle events.

    Important MPI rule:

    Every MPI rank must issue the same SPARTA command sequence in the same order.
    If one rank reloads the deck while the others only advance, the run can hang
    or appear to stall. The collective reload method below fixes that by forcing
    all ranks to make one shared reload decision before any SPARTA command is sent.
*/

namespace {

/*
    Small helper struct for optional shield diagnostics.

    The shield collision CSV is expected to contain the columns:
    step,time,shield_hits,reemitted_total

    Only the last non-empty data row is used.
*/
struct ShieldDiag {
    double shield_hits{0.0};
    double reemit_total{0.0};
};

/*
    Read the last data row from a simple CSV file containing shield collision data.

    Returns std::nullopt if:
    - the file does not exist
    - the file is empty
    - parsing fails badly enough that no useful row is available
*/
std::optional<ShieldDiag> read_shield_collide_csv(const fs::path& p) {
    std::ifstream in(p);
    if (!in) return std::nullopt;

    std::string line;
    std::string last;

    // Skip header, keep the last non-empty data line.
    if (!std::getline(in, line)) return std::nullopt;
    while (std::getline(in, line)) {
        if (!line.empty()) last = line;
    }
    if (last.empty()) return std::nullopt;

    std::stringstream ss(last);
    std::string token;

    // step
    if (!std::getline(ss, token, ',')) return std::nullopt;
    // time
    if (!std::getline(ss, token, ',')) return std::nullopt;

    ShieldDiag d;

    if (!std::getline(ss, token, ',')) return std::nullopt;
    try {
        d.shield_hits = std::stod(token);
    } catch (...) {
        d.shield_hits = 0.0;
    }

    if (!std::getline(ss, token, ',')) {
        d.reemit_total = 0.0;
    } else {
        try {
            d.reemit_total = std::stod(token);
        } catch (...) {
            d.reemit_total = 0.0;
        }
    }

    return d;
}

/*
    Per-process guard against duplicate tick logging.

    If WakeChamber::tick() is called more than once for the same outer tick
    on a given rank, the second call becomes a no-op for logging purposes.
*/
int& last_logged_tick_ref() {
    static int v = -1;
    return v;
}

/*
    Carry-forward storage for wake diagnostics.

    If a given tick does not produce a valid temperature or density reading,
    the most recent valid value is reused. This avoids contaminating logs
    with NaNs when the diagnostics file lags behind.
*/
double& last_tempK_ref() {
    static double v = 0.0;
    return v;
}

double& last_nrho_ref() {
    static double v = 0.0;
    return v;
}

/*
    Baseline free-stream density used to form a density ratio.

    The first valid density observed becomes the reference density.
*/
double& n_inf_ref() {
    static double v = 0.0;
    return v;
}

/*
    Remembered SPARTA block size.

    The wake chamber advances in repeated blocks of SPARTA steps.
    If the caller supplies a positive block size, it is remembered here.
*/
int& block_steps_ref() {
    static int v = 1000;
    return v;
}

/*
    In-memory state for params.inc.

    This lets setParameter() rewrite the whole params file every time while
    keeping previously assigned variables intact.
*/
std::map<std::string, double>& param_state_ref() {
    static std::map<std::string, double> m;
    return m;
}

} // namespace

WakeChamber::WakeChamber(MPI_Comm comm, std::string label)
    : comm_(comm), label_(std::move(label)) {}

WakeChamber::~WakeChamber() = default;

/*
    Initialize the wake chamber and load the SPARTA deck.

    deck_basename:
        SPARTA deck filename, for example "in.wake_harness"

    input_subdir:
        Input directory used by SpartaBridge to resolve the deck and related files

    Notes:
    - This method is idempotent. If already initialized, it returns immediately.
    - The SPARTA diagnostics path is set up here as well.
*/
void WakeChamber::init(const std::string& deck_basename,
                       const std::string& input_subdir) {
    if (initialized_) return;

    deck_ = deck_basename;
    input_subdir_ = input_subdir;

    // SPARTA writes diagnostics relative to the input directory.
    diag_path_ = fs::path(input_subdir_) / "data" / "tmp" / "wake_diag.csv";

    sp_ = std::make_unique<SpartaBridge>(comm_);
    sp_->runDeck(deck_, input_subdir_);

    initialized_ = true;
    dirtyReload_ = false;
    cum_steps_ = 0;
    last_run_steps_ = 0;

    logEvent_(/*status*/1.0,
              /*ran_steps*/0.0,
              /*cum_steps*/0.0,
              /*reload*/0.0,
              /*mark_reload*/0.0);
}

/*
    Advance one outer wake tick.

    nDefault:
        Requested SPARTA block size for this outer tick

    Behavior:
    - Remembers the first positive block size it sees
    - Always advances by the remembered block size once per outer tick
*/
void WakeChamber::step(int nDefault) {
    if (!initialized_) {
        throw std::runtime_error("WakeChamber::init() not called");
    }

    int& bs = block_steps_ref();
    if (nDefault > 0) {
        bs = nDefault;
    }

    if (bs > 0) {
        runSteps(bs);
    }
}

/*
    Send a raw SPARTA run command for n steps.

    This updates:
    - cum_steps_      total SPARTA steps executed over the lifetime of this object
    - last_run_steps_ most recent SPARTA block size that was actually run
*/
void WakeChamber::runSteps(int n) {
    if (!initialized_) {
        throw std::runtime_error("WakeChamber::init() not called");
    }

    if (n <= 0) {
        last_run_steps_ = 0;
        return;
    }

    sp_->command(("run " + std::to_string(n)).c_str());
    cum_steps_ += n;
    last_run_steps_ = n;
}

/*
    Mark the SPARTA state as needing a deck reload before the next advance.

    This only sets a local flag. The actual reload decision must be made
    collectively across MPI ranks before commands are issued.
*/
void WakeChamber::markDirtyReload() {
    dirtyReload_ = true;

    logEvent_(/*status*/1.0,
              /*ran_steps*/0.0,
              /*cum_steps*/static_cast<double>(cum_steps_),
              /*reload*/0.0,
              /*mark_reload*/1.0);
}

/*
    Original local reload-or-advance method.

    This is left here for compatibility, but it is not MPI-safe if different
    ranks disagree about dirtyReload_. In multi-rank runs, prefer the
    collective method below.
*/
bool WakeChamber::runIfDirtyOrAdvance(int n) {
    if (!initialized_) {
        throw std::runtime_error("WakeChamber::init() not called");
    }

    if (dirtyReload_) {
        sp_->command("clear");
        sp_->runDeck(deck_, input_subdir_);

        dirtyReload_ = false;
        last_run_steps_ = 0;

        logEvent_(/*status*/1.0,
                  /*ran_steps*/0.0,
                  /*cum_steps*/static_cast<double>(cum_steps_),
                  /*reload*/1.0,
                  /*mark_reload*/0.0);

        if (n > 0) runSteps(n);
        return true;
    }

    if (n > 0) runSteps(n);
    return (n > 0);
}

/*
    MPI-safe reload-or-advance method.

    Why this exists:
    In an MPI SPARTA run, every rank must issue the same command sequence.
    If rank 0 reloads while ranks 1 through N only advance, the simulation can
    hang or appear to stall.

    How it works:
    1. Each rank reports whether it thinks a reload is needed.
    2. MPI_Allreduce with MPI_MAX computes one shared answer.
       If any rank says reload is needed, all ranks reload.
    3. All ranks then execute the same SPARTA command sequence.

    Returns:
    - true if a positive number of steps was run
    - false if n <= 0 and no steps were run
*/
bool WakeChamber::runIfDirtyOrAdvanceCollective(int n) {
    if (!initialized_) {
        throw std::runtime_error("WakeChamber::runIfDirtyOrAdvanceCollective() not called after init()");
    }

    int local_dirty = dirtyReload_ ? 1 : 0;
    int global_dirty = 0;

    MPI_Allreduce(&local_dirty, &global_dirty, 1, MPI_INT, MPI_MAX, comm_);

    if (global_dirty) {
        /*
            Safety barrier before clear and reload.

            This gives rank 0 time to finish rewriting params.inc and ensures
            all ranks arrive at the same phase before issuing SPARTA commands.
        */
        MPI_Barrier(comm_);

        sp_->command("clear");
        sp_->runDeck(deck_, input_subdir_);

        dirtyReload_ = false;
        last_run_steps_ = 0;

        logEvent_(/*status*/1.0,
                  /*ran_steps*/0.0,
                  /*cum_steps*/static_cast<double>(cum_steps_),
                  /*reload*/1.0,
                  /*mark_reload*/0.0);

        if (n > 0) {
            runSteps(n);
            return true;
        }
        return false;
    }

    if (n > 0) {
        runSteps(n);
        return true;
    }

    return false;
}

/*
    Tear down the SPARTA bridge and reset internal state.
*/
void WakeChamber::shutdown() {
    logEvent_(/*status*/0.0,
              /*ran_steps*/0.0,
              /*cum_steps*/static_cast<double>(cum_steps_),
              /*reload*/0.0,
              /*mark_reload*/0.0);

    sp_.reset();
    initialized_ = false;
    dirtyReload_ = false;
    deck_.clear();
    input_subdir_.clear();
}

/*
    Rewrite params.inc with the latest parameter values.

    Behavior:
    - Only rank 0 writes the file
    - All known parameters are rewritten every time
    - A barrier ensures all ranks see a fully written file before continuing
*/
void WakeChamber::setParameter(const std::string& name, double value) {
    fs::path params = fs::path(PROJECT_SOURCE_DIR) / input_subdir_ / "params.inc";

    int rank = 0;
    MPI_Comm_rank(comm_, &rank);

    if (rank != 0) {
        return;
    }

    auto& state = param_state_ref();
    state[name] = value;

    std::ofstream out(params);
    if (!out) {
        throw std::runtime_error("WakeChamber::setParameter: cannot open params.inc");
    }

    for (const auto& kv : state) {
        out << "variable " << kv.first << " equal " << kv.second << "\n";
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("WakeChamber::setParameter: failed while writing params.inc");
    }

    Logger::instance().log(
        "Params",
        /*tick*/++event_id_,
        /*time*/0.0,
        {{label_ + ".param." + name, value}}
    );
}

/*
    Per-tick logging hook for wake diagnostics.

    Reads:
    - wake_diag.csv for temperature and density
    - shield_collide.csv for shield hit counters if available

    Logs:
    - status
    - most recent run block size
    - cumulative step count
    - derived pressure
    - density ratio relative to the first valid density
    - shield diagnostics

    This method does not perform SPARTA advancement.
*/
void WakeChamber::tick(const TickContext& ctx) {
    int& last_tick = last_logged_tick_ref();
    if (last_tick == ctx.tick_index) {
        last_run_steps_ = 0;
        return;
    }
    last_tick = ctx.tick_index;

    double temp_K = std::numeric_limits<double>::quiet_NaN();
    double density_m3 = std::numeric_limits<double>::quiet_NaN();

    if (auto diag = read_sparta_diag_csv(diag_path_)) {
        if (std::isfinite(diag->temp_K)) {
            temp_K = diag->temp_K;
        }
        if (std::isfinite(diag->density_m3)) {
            density_m3 = diag->density_m3;
        }
    }

    // Carry forward the last valid readings to reduce log discontinuities.
    double& lastT = last_tempK_ref();
    double& lastN = last_nrho_ref();

    if (!std::isfinite(temp_K)) {
        temp_K = lastT;
    } else {
        lastT = temp_K;
    }

    if (!std::isfinite(density_m3)) {
        density_m3 = lastN;
    } else {
        lastN = density_m3;
    }

    double& n_inf = n_inf_ref();
    if (n_inf <= 0.0 && density_m3 > 0.0) {
        n_inf = density_m3;
    }

    const double pressure_Pa =
        (temp_K > 0.0 && density_m3 > 0.0)
            ? (K_BOLTZ * temp_K * density_m3)
            : 0.0;

    const double n_ratio =
        (n_inf > 0.0 && density_m3 >= 0.0)
            ? (density_m3 / n_inf)
            : 0.0;

    const fs::path surf_csv = fs::path(input_subdir_) / "data" / "tmp" / "shield_collide.csv";

    double shield_hits = 0.0;
    double reemit_total = 0.0;

    if (auto sd = read_shield_collide_csv(surf_csv)) {
        shield_hits = sd->shield_hits;
        reemit_total = sd->reemit_total;
    }

    Logger::instance().log_wide(
        "WakeChamber",
        ctx.tick_index,
        ctx.time,
        {
            "status",
            "ran_steps",
            "cum_steps",
            "reload",
            "mark_reload",
            "temp_K",
            "density_m3",
            "n_ratio",
            "pressure_Pa",
            "shield_hits",
            "shield_reemit"
        },
        {
            1.0,
            static_cast<double>(last_run_steps_),
            static_cast<double>(cum_steps_),
            0.0,
            0.0,
            temp_K,
            density_m3,
            n_ratio,
            pressure_Pa,
            shield_hits,
            reemit_total
        }
    );

    last_run_steps_ = 0;
}

/*
    Log lifecycle and reload events.

    Only rank 0 writes these event rows so the event log is not duplicated
    across ranks.
*/
void WakeChamber::logEvent_(double status,
                            double ran_steps,
                            double cum_steps,
                            double reload,
                            double mark_reload) {
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank != 0) return;

    static const std::vector<std::string> COLS = {
        "status",
        "ran_steps",
        "cum_steps",
        "reload",
        "mark_reload"
    };

    const std::vector<double> vals = {
        status,
        ran_steps,
        cum_steps,
        reload,
        mark_reload
    };

    Logger::instance().log_wide(
        label_ + "Events",
        /*tick*/++event_id_,
        /*time*/0.0,
        COLS,
        vals
    );
}