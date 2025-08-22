#include "Logger.hpp"
#include <iostream>

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

Logger::~Logger() {
    if (central_.is_open()) central_.close();
    for (auto& kv : per_node_) {
        if (kv.second.is_open()) kv.second.close();
    }
}

void Logger::log(const std::string& subsystem,
                 int tick, double time,
                 const std::map<std::string, double>& values) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!central_.is_open()) {
        central_.open("../../data/raw/simulation.log", std::ios::out);
        central_ << "tick,time,subsystem,key,value\n";
    }

    if (!per_node_.count(subsystem)) {
        per_node_[subsystem].open("../../data/raw/" + subsystem + ".csv", std::ios::out);
        per_node_[subsystem] << "tick,time,key,value\n";
    }

    for (const auto& [k, v] : values) {
    central_ << tick << "," << time << "," << subsystem << "," << k << "," << v << "\n";
    per_node_[subsystem] << tick << "," << time << "," << k << "," << v << "\n";
    }
    central_.flush();
    per_node_[subsystem].flush();

}
