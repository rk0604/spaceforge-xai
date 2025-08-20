cmake_minimum_required(VERSION 3.16)
project(SimulationTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files for the simulation (adjust paths as needed)
set(SIM_SRC
    ../Sim/src/SimulationEngine.cpp
    ../Sim/src/TickPhaseEngine.cpp
    ../Sim/src/Logger.cpp
    ../Sim/src/Battery.cpp
    ../Sim/src/PowerBus.cpp
    ../Sim/src/SolarArray.cpp
    ../Sim/src/HeaterBank.cpp
    ../Sim/src/EffusionCell.cpp
)

# Build the test executable
add_executable(runTests tests.cpp ${SIM_SRC})

target_include_directories(runTests PRIVATE ../Sim/include)
target_link_libraries(runTests PRIVATE pthread)