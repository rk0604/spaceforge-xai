#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "Subsystem.hpp"
#include "TickContext.hpp"

class TickPhaseEngine {
public:
    TickPhaseEngine();
    ~TickPhaseEngine();

    void addSubsystem(Subsystem* s);
    void runTick(const TickContext& ctx);
    void start();   // NEW
    void stop();    // NEW

private:
    std::vector<Subsystem*> subsystems_;
    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};

    TickContext currentCtx_; // store full context
    std::atomic<int> tickIndex_{0};
    std::atomic<int> doneCount_{0};
};
