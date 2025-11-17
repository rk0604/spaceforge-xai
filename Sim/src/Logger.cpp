// Sim/src/Logger.cpp
#include "Logger.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <system_error>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;

// Resolve the base directory for logs.
//
// Priority:
//   1) env SF_LOG_DIR
//   2) <PROJECT_SOURCE_DIR>/data/raw
//   3) ./data/raw
//
// If env RUN_ID is set, we append it as a subdirectory so each run gets its
// own folder, e.g. data/raw/test_low_alt3/Battery.csv
fs::path resolve_base_dir() {
    // 1) explicit override
    if (const char* env = std::getenv("SF_LOG_DIR")) {
        if (*env) {
            fs::path p(env);
            if (const char* run = std::getenv("RUN_ID")) {
                if (*run) p /= run;
            }
            return p;
        }
    }

    // 2) project source dir if available
#ifdef PROJECT_SOURCE_DIR
    fs::path base = fs::path(PROJECT_SOURCE_DIR) / "data" / "raw";
#else
    // 3) fallback to current working directory
    fs::path base = fs::current_path() / "data" / "raw";
#endif

    if (const char* run = std::getenv("RUN_ID")) {
        if (*run) base /= run;
    }
    return base;
}

// Get or open the per-subsystem CSV file.
// If this is the first time we open it, we also write the header.
//
// For "tall" logs (log()), the header is: tick,time_s,key,value
// For "wide" logs (log_wide()), the header is: tick,time_s,<columns...>
std::ofstream& get_stream_for_subsystem(
    const std::string& subsystem,
    std::map<std::string, std::ofstream>& per_node,
    const std::vector<std::string>* wide_cols,
    bool is_wide
) {
    auto it = per_node.find(subsystem);
    if (it != per_node.end()) {
        return it->second;
    }

    fs::path base_dir = resolve_base_dir();
    std::error_code ec;
    fs::create_directories(base_dir, ec);
    if (ec) {
        throw std::runtime_error(
            "Logger: failed to create log directory " + base_dir.string() +
            " : " + ec.message()
        );
    }

    fs::path csv_path = base_dir / (subsystem + ".csv");
    std::ofstream out(csv_path, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "Logger: failed to open log file " + csv_path.string()
        );
    }

    // Write header
    if (is_wide) {
        out << "tick,time_s";
        if (wide_cols) {
            for (const auto& c : *wide_cols) {
                out << ',' << c;
            }
        }
        out << '\n';
    } else {
        out << "tick,time_s,key,value\n";
    }
    out.flush();

    auto [new_it, _] = per_node.emplace(subsystem, std::move(out));
    return new_it->second;
}

} // anonymous namespace

// ---------------- Logger public API ----------------

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (central_.is_open()) central_.close();
    for (auto& kv : per_node_) {
        if (kv.second.is_open()) kv.second.close();
    }
}

// Tall/long format: one row per (tick, key, value)
void Logger::log(const std::string& subsystem,
                 int tick, double time,
                 const std::map<std::string, double>& values) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::ofstream& out = get_stream_for_subsystem(
        subsystem,
        per_node_,
        /*wide_cols=*/nullptr,
        /*is_wide=*/false
    );

    for (const auto& kv : values) {
        out << tick << ',' << time << ','
            << kv.first << ',' << kv.second << '\n';
    }
    out.flush();
}

// Wide format: one row per tick with multiple named columns
void Logger::log_wide(const std::string& subsystem,
                      int tick, double time,
                      const std::vector<std::string>& cols,
                      const std::vector<double>& vals) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::ofstream& out = get_stream_for_subsystem(
        subsystem,
        per_node_,
        &cols,
        /*is_wide=*/true
    );

    out << tick << ',' << time;
    for (std::size_t i = 0; i < cols.size(); ++i) {
        double v = (i < vals.size() ? vals[i] : 0.0);
        out << ',' << v;
    }
    out << '\n';
    out.flush();
}
