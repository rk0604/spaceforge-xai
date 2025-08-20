#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"

int main() {
    // Create subsystems
    PowerBus bus;
    SolarArray solar;
    Battery battery;
    HeaterBank heater(200.0); // can draw up to 200 W

    // Wire connections
    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);

    // Setup simulation engine
    SimulationEngine engine;
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    engine.initialize();
    engine.setTickStep(0.1);

    // Run 50 ticks
    for (int i = 0; i < 50; ++i) {
        heater.setDemand(150.0);  // request power each tick
        engine.tick();            // 🚀 this dispatches ticks in parallel
    }

    engine.shutdown();
    return 0;
}
