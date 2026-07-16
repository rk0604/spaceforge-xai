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

/*
    Resolve the exact directory where subsystem CSV files must be written.

    The Slurm script passes OUTPUT_ROOT for every config and job.

    Example:
    OUTPUT_ROOT=/common/home/rvk22/spaceforge-xai-run2/data/raw/Config27/V4_job1

    The logger must use that exact folder.
    It must not append RUN_ID.
    It must not create data/raw/Config27_V4_job1.
    It must not create its own dataset folder.
*/
fs::path resolve_base_dir() {
    if (const char* output_root = std::getenv("OUTPUT_ROOT")) {
        if (*output_root) {
            return fs::path(output_root);
        }
    }

    if (const char* sf_log_dir = std::getenv("SF_LOG_DIR")) {
        if (*sf_log_dir) {
            return fs::path(sf_log_dir);
        }
    }

    return fs::current_path();
}

/*
    Escape one CSV field.

    Fields only get quotes when needed.
    Embedded quotes are doubled according to CSV rules.
*/
std::string escape_csv_field(const std::string& s) {
    bool needs_quotes = false;

    for (char ch : s) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return s;
    }

    std::string out;
    out.reserve(s.size() + 8);

    out.push_back('"');

    for (char ch : s) {
        if (ch == '"') {
            out.push_back('"');
            out.push_back('"');
        } else {
            out.push_back(ch);
        }
    }

    out.push_back('"');

    return out;
}

/*
    Open the CSV stream for one subsystem.

    Each subsystem gets one file:
    Battery.csv
    EffusionCell.csv
    HeaterBank.csv
    Orbit.csv
    PowerBus.csv
    ProcessState.csv
    ScheduleState.csv
    SimulationEngine.csv
    SolarArray.csv
    substrate.csv
*/
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
            "Logger failed to create output directory "
            + base_dir.string()
            + " : "
            + ec.message()
        );
    }

    fs::path csv_path = base_dir / (subsystem + ".csv");

    std::ofstream out(csv_path, std::ios::out | std::ios::trunc);

    if (!out) {
        throw std::runtime_error(
            "Logger failed to open output file "
            + csv_path.string()
        );
    }

    if (is_wide) {
        out << "tick,time_s";

        if (wide_cols) {
            for (const auto& c : *wide_cols) {
                out << ',' << escape_csv_field(c);
            }
        }

        out << '\n';
    } else {
        out << "tick,time_s,key,value\n";
    }

    out.flush();

    auto inserted = per_node.emplace(subsystem, std::move(out));

    return inserted.first->second;
}

} // anonymous namespace

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (central_.is_open()) {
        central_.close();
    }

    for (auto& kv : per_node_) {
        if (kv.second.is_open()) {
            kv.second.close();
        }
    }
}

void Logger::log(const std::string& subsystem,
                 int tick,
                 double time,
                 const std::map<std::string, double>& values) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::ofstream& out = get_stream_for_subsystem(
        subsystem,
        per_node_,
        nullptr,
        false
    );

    for (const auto& kv : values) {
        out << tick << ','
            << time << ','
            << escape_csv_field(kv.first) << ','
            << kv.second << '\n';
    }

    out.flush();
}

void Logger::log_wide(const std::string& subsystem,
                      int tick,
                      double time,
                      const std::vector<std::string>& cols,
                      const std::vector<double>& vals) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::ofstream& out = get_stream_for_subsystem(
        subsystem,
        per_node_,
        &cols,
        true
    );

    out << tick << ',' << time;

    for (std::size_t i = 0; i < cols.size(); ++i) {
        double v = 0.0;

        if (i < vals.size()) {
            v = vals[i];
        }

        out << ',' << v;
    }

    out << '\n';
    out.flush();
}

void Logger::log_wide(const std::string& subsystem,
                      int tick,
                      double time,
                      const std::vector<std::string>& cols,
                      const std::vector<std::string>& vals) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::ofstream& out = get_stream_for_subsystem(
        subsystem,
        per_node_,
        &cols,
        true
    );

    out << tick << ',' << time;

    for (std::size_t i = 0; i < cols.size(); ++i) {
        std::string v;

        if (i < vals.size()) {
            v = vals[i];
        }

        out << ',' << escape_csv_field(v);
    }

    out << '\n';
    out.flush();
}