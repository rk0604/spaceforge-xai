#pragma once
#include <fstream>
#include <string>
#include <mutex>
#include <map>
#include <memory>

class Logger {
public:
    static Logger& instance();

    void log(const std::string& subsystem,
             int tick, double time,
             const std::map<std::string, double>& values);

private:
    Logger() = default;
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex mtx_;
    std::ofstream central_;
    std::map<std::string, std::ofstream> per_node_;
};
