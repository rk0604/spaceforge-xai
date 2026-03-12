#pragma once

#include <mutex>
#include <map>
#include <string>
#include <fstream>
#include <vector>

class Logger {
public:
    static Logger& instance();
    ~Logger();

    // Tall/long format: one row per key
    void log(const std::string& subsystem,
             int tick, double time,
             const std::map<std::string, double>& values);

    // Wide format: one row per call with multiple numeric columns.
    //
    // This is the EXISTING API and remains unchanged so no existing loggers break.
    void log_wide(const std::string& subsystem,
                  int tick, double time,
                  const std::vector<std::string>& columns,
                  const std::vector<double>& values);

    // Wide format overload for string/text columns.
    //
    // This is additive and does not affect any existing numeric call sites.
    // Use this when you want readable CSV values such as:
    //   scheduler_state_name = "warming"
    //   prep_mode_name       = "cooldown"
    void log_wide(const std::string& subsystem,
                  int tick, double time,
                  const std::vector<std::string>& columns,
                  const std::vector<std::string>& values);

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex mtx_;
    std::ofstream central_;
    std::map<std::string, std::ofstream> per_node_;
};