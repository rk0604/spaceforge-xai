cmake_minimum_required(VERSION 3.20)
project(SpaceForgeSim LANGUAGES CXX)

# -------------------- Dependencies --------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# Option: embed SPARTA (OFF by default; external spa_ via --sparta-exe)
option(ENABLE_SPARTA "Enable in-process SPARTA coupling (requires SPARTA headers/lib)" OFF)
set(SPARTA_DIR "$ENV{SPARTA_DIR}" CACHE PATH "SPARTA source/build root (has src/, libsparta*.a)")

# -------------------- Sources --------------------
set(SIMCORE_SOURCES
  src/Logger.cpp
  src/SimulationEngine.cpp
  src/TickPhaseEngine.cpp
  src/PowerBus.cpp
  src/Battery.cpp
  src/SolarArray.cpp
  src/EffusionCell.cpp
  src/WakeChamber.cpp
  src/DepositionMap.cpp
  src/SpartaDiag.cpp
  src/orbit.cpp
  src/GrowthMonitor.cpp
  src/helpers.cpp
  # Add HeaterBank implementation:
  src/HeaterBank.cpp
)

set(SIMCORE_HEADERS
  include/Logger.hpp
  include/TickContext.hpp
  include/SimulationEngine.hpp
  include/TickPhaseEngine.hpp
  include/PowerBus.hpp
  include/orbit.hpp
  include/GrowthMonitor.hpp
  include/Battery.hpp
  include/SolarArray.hpp
  include/EffusionCell.hpp
  include/WakeChamber.hpp
  include/DepositionMap.hpp
  include/SpartaDiag.hpp
  include/helpers.hpp
  include/SpartaBridge.hpp
  # Add HeaterBank header:
  include/HeaterBank.hpp
)

# Choose Sparta bridge implementation
if(ENABLE_SPARTA)
  list(APPEND SIMCORE_SOURCES src/SpartaBridge.cpp)
  include_directories(${SPARTA_DIR} ${SPARTA_DIR}/src)
else()
  list(APPEND SIMCORE_SOURCES src/SpartaBridgeShim.cpp)
endif()

add_library(simcore STATIC ${SIMCORE_SOURCES} ${SIMCORE_HEADERS})
target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simcore PUBLIC MPI::MPI_CXX)

target_compile_definitions(simcore PRIVATE
  ENABLE_SPARTA=$<BOOL:${ENABLE_SPARTA}>
  PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# If embedding, try to link a SPARTA lib/target only if it exists
if(ENABLE_SPARTA)
  if(TARGET sparta_static)
    target_link_libraries(simcore PUBLIC sparta_static)
  else()
    message(FATAL_ERROR "ENABLE_SPARTA=ON but no SPARTA CMake target/library found. Build no-embed or provide sparta_static/libsparta.")
  endif()
endif()

# -------------------- Executable: sim --------------------
option(BUILD_SIM_APP "Build the Sim executable" ON)
if(BUILD_SIM_APP)
  add_executable(sim src/main.cpp)
  target_link_libraries(sim PRIVATE simcore)
  set_target_properties(sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Sim")

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(sim PRIVATE "-Wl,--no-undefined")
  endif()
endif()
