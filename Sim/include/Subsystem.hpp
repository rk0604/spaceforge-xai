#pragma once
#include <string>
#include "TickContext.hpp"

class Subsystem {
public:
    explicit Subsystem(const std::string& name) : name_(name) {}
    virtual ~Subsystem() = default;

    // lifecycle
    virtual void initialize() = 0;
    virtual void tick(const TickContext& ctx) = 0;
    virtual void shutdown() = 0;

    // access
    std::string getName() const { return name_; }

protected:
    std::string name_;
};
