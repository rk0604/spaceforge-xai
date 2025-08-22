#include "TickPhaseEngine.hpp"

TickPhaseEngine::TickPhaseEngine()
    : running_(false),
      tickIndex_(-1),
      doneCount_(0) {}


TickPhaseEngine::~TickPhaseEngine() {
    stop();
}


void TickPhaseEngine::start() {
    running_ = true;
    for (auto* s : subsystems_) {
        threads_.emplace_back([&, s]() {
            int seen = -1;
            while (true) {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { 
                    return !running_ || tickIndex_ > seen; 
                });

                if (!running_) break;

                seen = tickIndex_;
                TickContext ctx = currentCtx_;  // guaranteed valid now
                lock.unlock();

                s->tick(ctx);
                doneCount_.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
}


void TickPhaseEngine::stop() {
    if (!running_) return;
    running_ = false;
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

void TickPhaseEngine::addSubsystem(Subsystem* s) {
    subsystems_.push_back(s);
    // Worker threads will be spawned in start()
}

void TickPhaseEngine::runTick(const TickContext& ctx) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        tickIndex_   = ctx.tick_index;
        currentCtx_  = ctx;
        doneCount_   = 0;
    }

    cv_.notify_all();

    while (doneCount_.load() < subsystems_.size()) {
        std::this_thread::yield();
    }
}
