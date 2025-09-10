#include "Logger.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <system_error>
#include <mutex>
#include <string>
#include <vector>     // <-- needed

namespace {
namespace fs = std::filesystem;

// Resolve the base directory for logs.
// Priority: env(SF_LOG_DIR) → <PROJECT_SOURCE_DIR>/data/raw → ./data/raw
inline fs::path log_base_dir() {
    if (const char* env = std::getenv("SF_LOG_DIR"); env && *env) {
        return fs::path(env);
    }
#ifdef PROJECT_SOURCE_DIR
    return fs::path(PROJECT_SOURCE_DIR) / "data" / "raw";
#else
    return fs::path("data") / "raw";
#endif
}

// Ensure directory exists (best-effort; no throw)
inline void ensure_dir(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
    if (ec) {
        std::cerr << "[Logger] Warning: failed to create log dir " << p
                  << " : " << ec.message() << "\n";
    }
}

inline void warn_open_failure(const fs::path& p) {
    std::error_code ec;
    auto abs = fs::absolute(p, ec);
    std::cerr << "[Logger] Failed to open " << (ec ? p : abs)
              << " (cwd=" << fs::current_path() << ")\n";
}

} // namespace

// -----------------------------------------------------------------------------
// Singleton
// -----------------------------------------------------------------------------
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

// -----------------------------------------------------------------------------
// Dtor: close any open streams
// -----------------------------------------------------------------------------
Logger::~Logger() {
    if (central_.is_open()) central_.close();
    for (auto& kv : per_node_) {
        if (kv.second.is_open()) kv.second.close();
    }
}

// -----------------------------------------------------------------------------
// Log one batch of values for a subsystem at (tick, time) in long format
// -----------------------------------------------------------------------------
void Logger::log(const std::string& subsystem,
                 int tick, double time,
                 const std::map<std::string, double>& values) {
    std::lock_guard<std::mutex> lock(mtx_);

    const fs::path base = log_base_dir();
    ensure_dir(base);

    // Open central log once
    if (!central_.is_open()) {
        const fs::path central_path = base / "simulation.log";
        central_.open(central_path.string(), std::ios::out);
        if (!central_) {
            warn_open_failure(central_path);
        } else {
            central_ << "tick,time_s,subsystem,key,value\n";
            central_.flush();
        }
    }

    // Open per-subsystem CSV once
    if (!per_node_.count(subsystem)) {
        const fs::path node_path = base / (subsystem + ".csv");
        auto& ofs = per_node_[subsystem];
        ofs.open(node_path.string(), std::ios::out);
        if (!ofs) {
            warn_open_failure(node_path);
        } else {
            ofs << "tick,time_s,key,value\n";
            ofs.flush();
        }
    }

    // Write rows
    auto write_row = [&](std::ostream& os, const std::string& key, double val) {
        os << tick << ',' << time << ',';
        if (&os == &central_) {
            os << subsystem << ',' << key << ',' << val << '\n';
        } else {
            os << key << ',' << val << '\n';
        }
    };

    for (const auto& [k, v] : values) {
        if (central_) write_row(central_, k, v);
        auto it = per_node_.find(subsystem);
        if (it != per_node_.end() && it->second) write_row(it->second, k, v);
    }

    if (central_) central_.flush();
    auto it = per_node_.find(subsystem);
    if (it != per_node_.end() && it->second) it->second.flush();
}

// -----------------------------------------------------------------------------
// Wide format: one row with multiple columns
// -----------------------------------------------------------------------------
void Logger::log_wide(const std::string& subsystem, int tick, double time,
                      const std::vector<std::string>& cols,
                      const std::vector<double>& vals) {
    std::lock_guard<std::mutex> lock(mtx_);

    const fs::path base = log_base_dir();
    ensure_dir(base);

    // Open per-subsystem CSV once
    auto& out = per_node_[subsystem];
    if (!out.is_open()) {
        const fs::path node_path = base / (subsystem + ".csv");
        out.open(node_path.string(), std::ios::out);
        if (!out) {
            warn_open_failure(node_path);
            return;
        }
        // write header once: tick,time,<cols...>
        out << "tick,time_s";
        for (const auto& c : cols) out << ',' << c;
        out << '\n';
        out.flush();
    }

    // Write the row
    out << tick << ',' << time;
    for (size_t i = 0; i < cols.size(); ++i) {
        out << ',' << (i < vals.size() ? vals[i] : 0.0);
    }
    out << '\n';
    out.flush();
}
