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

    // Wide format: one row per call with multiple columns
    void log_wide(const std::string& subsystem,
                  int tick, double time,
                  const std::vector<std::string>& columns,
                  const std::vector<double>& values);

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex mtx_;
    std::ofstream central_;
    std::map<std::string, std::ofstream> per_node_;
};
