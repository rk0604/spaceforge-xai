#include "EffusionCell.hpp"
#include "Logger.hpp"

void EffusionCell::applyHeat(double watts, double dt) {
    // Simple dummy model: heat increases temperature linearly with power
    double dT = watts * dt * 0.01;   // fake coefficient
    temperature_ += dT;

    Logger::instance().log("EffusionCell", 0, 0.0,
        {{"heatInput", watts}, {"temperature", temperature_}});
}
