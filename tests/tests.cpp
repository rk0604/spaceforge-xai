#include "SimulationEngine.hpp"
#include "Battery.hpp"
#include "SolarArray.hpp"
#include "PowerBus.hpp"
#include "HeaterBank.hpp"
#include <cassert>
#include <iostream>

// ✅ Helper: run a short simulation
void run_basic_simulation(int ticks, double step) {
    PowerBus bus;
    SolarArray solar;
    Battery battery;
    HeaterBank heater(200.0);

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);

    SimulationEngine engine;
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    engine.initialize();
    engine.setTickStep(step);

    for (int i = 0; i < ticks; i++) {
        heater.setDemand(150.0);
        engine.tick();
    }

    engine.shutdown();
    std::cout << "[TEST] Completed " << ticks << " ticks.\n";
}

// ✅ Test 1: Basic sanity (runs without segfaults)
void test_basic_run() {
    run_basic_simulation(10, 0.1);
    std::cout << "[PASS] Basic run executed.\n";
}

// ✅ Test 2: Battery bounds
void test_battery_bounds() {
    PowerBus bus;
    SolarArray solar;
    Battery battery;
    HeaterBank heater(200.0);

    solar.setPowerBus(&bus);
    battery.setPowerBus(&bus);
    heater.setPowerBus(&bus);

    SimulationEngine engine;
    engine.addSubsystem(&solar);
    engine.addSubsystem(&bus);
    engine.addSubsystem(&battery);
    engine.addSubsystem(&heater);

    engine.initialize();
    engine.setTickStep(0.1);

    for (int i = 0; i < 100; i++) {
        heater.setDemand(150.0);
        engine.tick();
        double c = battery.getCharge();
        assert(c >= 0.0 && c <= 1000.0); // ✅ stays within bounds
    }

    engine.shutdown();
    std::cout << "[PASS] Battery stayed within [0, capacity].\n";
}

// ✅ Test 3: PowerBus never negative
void test_bus_never_negative() {
    PowerBus bus;
    bus.initialize();
    bus.addPower(100);

    TickContext ctx {0, 0.0, 0.1};

    double draw = bus.drawPower(150, ctx); // should not overdraw
    assert(draw <= 100);
    assert(bus.getAvailablePower() >= 0);
    std::cout << "[PASS] PowerBus never went negative.\n";
}

int main() {
    test_basic_run();
    test_battery_bounds();
    test_bus_never_negative();
    std::cout << "✅ All tests passed.\n";
    return 0;
}
