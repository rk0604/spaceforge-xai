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
#include <mpi.h>

namespace fs = std::filesystem;

// ---------- helpers (file-local) ----------

namespace {

struct ShieldDiag {
    double shield_hits{0.0};
    double reemit_total{0.0};
};

// Read the *last* data line of a simple CSV and parse
// Expected columns: step,time,shield_hits,reemitted_total
// Returns std::nullopt if file is missing/empty.
std::optional<ShieldDiag> read_shield_collide_csv(const fs::path& p) {
    std::ifstream in(p);
    if (!in) return std::nullopt;

    std::string line, last;
    // skip header, capture last non-empty line
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
    try { d.shield_hits = std::stod(token); } catch (...) { d.shield_hits = 0.0; }

    if (!std::getline(ss, token, ',')) { d.reemit_total = 0.0; }
    else {
        try { d.reemit_total = std::stod(token); } catch (...) { d.reemit_total = 0.0; }
    }
    return d;
}

// Per-process guard to avoid duplicate tick logging if multiple calls occur on a rank
int& last_logged_tick_ref() {
    static int v = -1;
    return v;
}

// Carry-forward last valid wake diagnostics to avoid NaNs
double& last_tempK_ref()   { static double v = 0.0; return v; }
double& last_nrho_ref()    { static double v = 0.0; return v; }
// Baseline n_infinity for density ratio; captured from first valid reading
double& n_inf_ref()        { static double v = 0.0; return v; }

// ---- NEW: remembered SPARTA steps per outer tick (default 1000) ----
int& block_steps_ref() {
    static int v = 1000;   // default if caller never supplies --sparta-block
    return v;
}

} // namespace


// ---------- WakeChamber implementation ----------

WakeChamber::WakeChamber(MPI_Comm comm, std::string label)
  : comm_(comm), label_(std::move(label)) {}

WakeChamber::~WakeChamber() = default;

void WakeChamber::init(const std::string& deck_basename,
                       const std::string& input_subdir) {
    if (initialized_) return;

    deck_         = deck_basename;
    input_subdir_ = input_subdir;

    // Resolve SPARTA diag path relative to project root (process CWD)
    // SPARTA runs with CWD at <PROJECT>/<input_subdir_>, and writes to data/tmp/...
    diag_path_ = fs::path(input_subdir_) / "data" / "tmp" / "wake_diag.csv";

    sp_ = std::make_unique<SpartaBridge>(comm_);
    sp_->runDeck(deck_, input_subdir_);

    initialized_     = true;
    dirtyReload_     = false;
    cum_steps_       = 0;
    last_run_steps_  = 0;

    // Event log: came online
    logEvent_(/*status*/1.0, /*ran_steps*/0.0, /*cum_steps*/0.0,
              /*reload*/0.0, /*mark_reload*/0.0);
}

void WakeChamber::step(int nDefault) {
    if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");

    // ---- NEW: remember the first positive nDefault (i.e., --sparta-block) ----
    int& bs = block_steps_ref();
    if (nDefault > 0) bs = nDefault;

    // Always advance by the remembered block size once per outer tick
    if (bs > 0) runSteps(bs);
}

void WakeChamber::runSteps(int n) {
    if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");
    if (n <= 0) { last_run_steps_ = 0; return; }

    sp_->command(("run " + std::to_string(n)).c_str());
    cum_steps_      += n;
    last_run_steps_  = n;
}

void WakeChamber::markDirtyReload() {
    dirtyReload_ = true;
    logEvent_(/*status*/1.0, /*ran_steps*/0.0, /*cum_steps*/(double)cum_steps_,
              /*reload*/0.0, /*mark_reload*/1.0);
}

bool WakeChamber::runIfDirtyOrAdvance(int n) {
    if (!initialized_) throw std::runtime_error("WakeChamber::init() not called");

    if (dirtyReload_) {
        sp_->command("clear");
        sp_->runDeck(deck_, input_subdir_);
        dirtyReload_    = false;
        last_run_steps_ = 0;
        logEvent_(/*status*/1.0, /*ran_steps*/0.0, /*cum_steps*/(double)cum_steps_,
                  /*reload*/1.0, /*mark_reload*/0.0);

        if (n > 0) runSteps(n);
        return true;
    } else {
        if (n > 0) runSteps(n);
        return (n > 0);
    }
}

void WakeChamber::shutdown() {
    logEvent_(/*status*/0.0, /*ran_steps*/0.0, /*cum_steps*/(double)cum_steps_,
              /*reload*/0.0, /*mark_reload*/0.0);

    sp_.reset();
    initialized_ = false;
    dirtyReload_ = false;
    deck_.clear();
    input_subdir_.clear();
}

void WakeChamber::setParameter(const std::string& name, double value) {
    fs::path params = fs::path(PROJECT_SOURCE_DIR) / input_subdir_ / "params.inc";

    int rank = 0; MPI_Comm_rank(comm_, &rank);
    if (rank == 0) {
        std::ofstream out(params);
        if (!out) throw std::runtime_error("WakeChamber::setParameter: cannot open params.inc");
        out << "variable " << name << " equal " << value << "\n";
        Logger::instance().log("Params", /*tick*/++event_id_, /*time*/0.0,
                               {{label_ + ".param." + name, value}});
    }
    MPI_Barrier(comm_);
}

void WakeChamber::tick(const TickContext& ctx) {
    // De-dupe per process (rank). We do NOT gate on rank==0, so any wake rank that calls tick() will log.
    int& last_tick = last_logged_tick_ref();
    if (last_tick == ctx.tick_index) {
        last_run_steps_ = 0;
        return;
    }
    last_tick = ctx.tick_index;

    // ---- Read wake diag (density/temp) ----
    double temp_K     = std::numeric_limits<double>::quiet_NaN();
    double density_m3 = std::numeric_limits<double>::quiet_NaN();

    if (auto diag = read_sparta_diag_csv(diag_path_)) {
        if (std::isfinite(diag->temp_K))     temp_K     = diag->temp_K;
        if (std::isfinite(diag->density_m3)) density_m3 = diag->density_m3;
    }

    // Carry-forward to avoid NaNs
    double& lastT = last_tempK_ref();
    double& lastN = last_nrho_ref();
    if (!std::isfinite(temp_K))     temp_K     = lastT; else lastT = temp_K;
    if (!std::isfinite(density_m3)) density_m3 = lastN; else lastN = density_m3;

    // Establish n_infinity the first time we see a valid density
    double& n_inf = n_inf_ref();
    if (n_inf <= 0.0 && density_m3 > 0.0) n_inf = density_m3;

    // Derived quantities
    const double pressure_Pa = (temp_K > 0.0 && density_m3 > 0.0)
        ? (K_BOLTZ * temp_K * density_m3)
        : 0.0;

    const double n_ratio = (n_inf > 0.0 && density_m3 >= 0.0)
        ? (density_m3 / n_inf)
        : 0.0;

    // ---- Read shield collision counters (if deck writes them) ----
    const fs::path surf_csv = fs::path(input_subdir_) / "data" / "tmp" / "shield_collide.csv";
    double shield_hits   = 0.0;
    double reemit_total  = 0.0;
    if (auto sd = read_shield_collide_csv(surf_csv)) {
        shield_hits  = sd->shield_hits;
        reemit_total = sd->reemit_total;
    }

    // ---- Log one wide row per tick ----
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
            /*reload flag this tick*/ 0.0,   // reload events are logged in Events file
            /*mark_reload*/           0.0,   // ditto
            temp_K,
            density_m3,
            n_ratio,
            pressure_Pa,
            shield_hits,
            reemit_total
        }
    );

    // consume the per-advance counter
    last_run_steps_ = 0;
}

void WakeChamber::logEvent_(double status, double ran_steps, double cum_steps,
                            double reload, double mark_reload) {
    int rank = 0; MPI_Comm_rank(comm_, &rank);
    if (rank != 0) return;

    static const std::vector<std::string> COLS = {
        "status", "ran_steps", "cum_steps", "reload", "mark_reload"
    };
    const std::vector<double> vals = {
        status, ran_steps, cum_steps, reload, mark_reload
    };

    Logger::instance().log_wide(label_ + "Events",
                                /*tick*/++event_id_, /*time*/0.0,
                                COLS, vals);
}
