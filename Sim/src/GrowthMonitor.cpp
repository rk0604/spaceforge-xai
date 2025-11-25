#include "GrowthMonitor.hpp"
#include "PowerBus.hpp"   // for drawPower()

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace {
namespace fs = std::filesystem;

// Resolve output directory for GrowthMonitor CSVs.
// Priority:
//   1) $SF_LOG_DIR/{RUN_ID}
//   2) PROJECT_SOURCE_DIR/data/raw/{RUN_ID}
//   3) ./data/raw/{RUN_ID}
fs::path resolve_growthmonitor_base_dir() {
    // Check SF_LOG_DIR override (same behavior as Logger)
    if (const char* env = std::getenv("SF_LOG_DIR")) {
        if (*env) {
            fs::path p(env);
            if (const char* run = std::getenv("RUN_ID")) {
                if (*run) p /= run;
            }
            return p;
        }
    }

    // If PROJECT_SOURCE_DIR available, use that
#ifdef PROJECT_SOURCE_DIR
    fs::path base = fs::path(PROJECT_SOURCE_DIR) / "data" / "raw";
#else
    // Fallback to CWD/data/raw
    fs::path base = fs::current_path() / "data" / "raw";
#endif

    if (const char* run = std::getenv("RUN_ID")) {
        if (*run) base /= run;
    }

    return base;
}
} // anonymous namespace


// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
GrowthMonitor::GrowthMonitor(int gridN)
    : Subsystem("GrowthMonitor"),
      gridN_(gridN),
      waferRadiusCells_(0.5 * static_cast<double>(gridN) * 0.95)  // inside edge
{
}


// -----------------------------------------------------------------------------
// Job setup
// -----------------------------------------------------------------------------
void GrowthMonitor::setNumJobs(std::size_t nJobs) {
    jobs_.clear();
    jobs_.resize(nJobs);
    ensureJobStorage();
}

void GrowthMonitor::setBeamState(int jobIndex, bool mbeOn, double Fwafer_cm2s) {
    activeJobIndex_    = jobIndex;
    mbeOn_             = mbeOn;
    currentFwaferCm2s_ = Fwafer_cm2s;
}

void GrowthMonitor::markJobAborted(int jobIndex) {
    if (jobIndex < 0) return;
    if (static_cast<std::size_t>(jobIndex) >= jobs_.size()) return;
    jobs_[static_cast<std::size_t>(jobIndex)].aborted = true;
}


// -----------------------------------------------------------------------------
// initialize(): now resolves CSV path into data/raw/{RUN_ID}
// -----------------------------------------------------------------------------
void GrowthMonitor::initialize() {
    if (!isLeader_) {
        return;
    }

    if (gridN_ <= 0) {
        gridN_ = 32;
    }

    buildWaferMask();
    ensureJobStorage();

    const char* env_run_id = std::getenv("RUN_ID");
    std::string run_id     = env_run_id ? env_run_id : "norunid";

    // Determine directory: follows Logger semantics
    fs::path base_dir = resolve_growthmonitor_base_dir();

    std::error_code ec;
    fs::create_directories(base_dir, ec);
    if (ec) {
        std::cerr << "[GrowthMonitor] Failed to create directory "
                  << base_dir << " : " << ec.message() << "\n";
    }

    // Full path: data/raw/{RUN_ID}/GrowthMonitor_{RUN_ID}.csv
    csvFilename_ = (base_dir / ("GrowthMonitor_" + run_id + ".csv")).string();
}


// -----------------------------------------------------------------------------
// tick()
// -----------------------------------------------------------------------------
void GrowthMonitor::tick(const TickContext& ctx) {
    if (!isLeader_) return;

    if (activeJobIndex_ < 0) return;
    if (!mbeOn_) return;
    if (!std::isfinite(currentFwaferCm2s_) || currentFwaferCm2s_ <= 0.0) return;
    if (static_cast<std::size_t>(activeJobIndex_) >= jobs_.size()) return;

    // Draw monitor power (not tracked for now)
    if (bus_ && monitorPowerW_ > 0.0) {
        (void)bus_->drawPower(monitorPowerW_, ctx);
    }

    integrateDose(ctx.dt, ctx.time);
}


// -----------------------------------------------------------------------------
// shutdown() — writes the CSV
// -----------------------------------------------------------------------------
void GrowthMonitor::shutdown() {
    if (!isLeader_) return;
    writeCsv();
}


// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------
void GrowthMonitor::buildWaferMask() {
    waferMask_.assign(static_cast<std::size_t>(gridN_ * gridN_), 0);

    const double cx = 0.5 * static_cast<double>(gridN_ - 1);
    const double cy = 0.5 * static_cast<double>(gridN_ - 1);

    for (int r = 0; r < gridN_; ++r) {
        for (int c = 0; c < gridN_; ++c) {
            const double x = static_cast<double>(c);
            const double y = static_cast<double>(r);
            const double dx = x - cx;
            const double dy = y - cy;
            const double dist = std::sqrt(dx * dx + dy * dy);

            const std::size_t idx = static_cast<std::size_t>(r * gridN_ + c);
            waferMask_[idx] = (dist <= waferRadiusCells_) ? 1u : 0u;
        }
    }
}

void GrowthMonitor::ensureJobStorage() {
    const std::size_t totalCells = static_cast<std::size_t>(gridN_) *
                                   static_cast<std::size_t>(gridN_);
    for (std::size_t j = 0; j < jobs_.size(); ++j) {
        jobs_[j].dose.assign(totalCells, 0.0);
        jobs_[j].had_growth   = false;
        jobs_[j].last_t_end_s = 0.0;
        jobs_[j].aborted      = false;
    }
}

void GrowthMonitor::integrateDose(double dt, double t_now) {
    if (dt <= 0.0) return;

    JobAccum& job = jobs_[static_cast<std::size_t>(activeJobIndex_)];

    const double increment = currentFwaferCm2s_ * dt;
    const std::size_t totalCells = static_cast<std::size_t>(gridN_) *
                                   static_cast<std::size_t>(gridN_);

    for (std::size_t idx = 0; idx < totalCells; ++idx) {
        if (!waferMask_[idx]) continue;
        job.dose[idx] += increment;
    }

    job.had_growth   = true;
    job.last_t_end_s = t_now;
}


// -----------------------------------------------------------------------------
// writeCsv()
// -----------------------------------------------------------------------------
void GrowthMonitor::writeCsv() const {
    if (csvFilename_.empty()) return;

    std::ofstream out(csvFilename_, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "[GrowthMonitor] Failed to open " << csvFilename_
                  << " for writing.\n";
        return;
    }

    out << "job_index,wafer_index,row,col,t_end_s,dose_arb\n";

    const std::size_t totalCells = static_cast<std::size_t>(gridN_) *
                                   static_cast<std::size_t>(gridN_);

    for (std::size_t j = 0; j < jobs_.size(); ++j) {
        const JobAccum& job = jobs_[j];
        if (!job.had_growth) continue;

        const double t_end = job.last_t_end_s;
        const int waferIndex = 0;

        for (int r = 0; r < gridN_; ++r) {
            for (int c = 0; c < gridN_; ++c) {
                const std::size_t idx = static_cast<std::size_t>(r * gridN_ + c);
                if (!waferMask_[idx]) continue;

                const double val = job.dose[idx];
                out << static_cast<int>(j) << ","
                    << waferIndex << ","
                    << r << ","
                    << c << ","
                    << t_end << ","
                    << val << "\n";
            }
        }
    }

    out.close();
}
