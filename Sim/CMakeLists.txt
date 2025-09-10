cmake_minimum_required(VERSION 3.20)

# MPI (already found in root, but fine to require the component here)
find_package(MPI REQUIRED COMPONENTS CXX)

# If someone builds Sim/ standalone, create/import SPARTA only if missing
if(NOT TARGET sparta_static)
  option(ENABLE_SPARTA "Enable SPARTA coupling" OFF)
  set(SPARTA_DIR "$ENV{SPARTA_DIR}" CACHE PATH "SPARTA build/install dir")

  if(ENABLE_SPARTA)
    find_path(SPARTA_INCLUDE_DIR
      NAMES sparta.h
      PATHS ${SPARTA_DIR} ${SPARTA_DIR}/src ${SPARTA_DIR}/include)
    find_library(SPARTA_LIBRARY
      NAMES sparta sparta_mpi sparta_static
      PATHS ${SPARTA_DIR} ${SPARTA_DIR}/lib ${SPARTA_DIR}/src)

    if(NOT SPARTA_INCLUDE_DIR OR NOT SPARTA_LIBRARY)
      message(FATAL_ERROR "Could not find SPARTA at SPARTA_DIR='${SPARTA_DIR}'. Set -DSPARTA_DIR=/path/to/sparta.")
    endif()

    add_library(sparta_static UNKNOWN IMPORTED)
    set_target_properties(sparta_static PROPERTIES
      IMPORTED_LOCATION "${SPARTA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${SPARTA_INCLUDE_DIR}")
  else()
    add_library(sparta_static INTERFACE)
  endif()
endif()

# ---- Library: simcore ----
set(SIMCORE_SOURCES
  src/Battery.cpp
  src/PowerBus.cpp
  src/SolarArray.cpp
  src/HeaterBank.cpp
  src/SimulationEngine.cpp
  src/TickPhaseEngine.cpp
  src/Logger.cpp
  src/SpartaDiag.cpp
  # Files that depend on SPARTA are only built when enabled:
)

if(ENABLE_SPARTA)
  list(APPEND SIMCORE_SOURCES
    src/SpartaBridge.cpp
    src/WakeChamber.cpp
    src/EffusionCell.cpp
    src/DepositionMap.cpp
  )
endif()

add_library(simcore ${SIMCORE_SOURCES})
target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simcore PUBLIC MPI::MPI_CXX sparta_static)
target_compile_definitions(simcore PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---- Executable: sim ----
option(BUILD_SIM_APP "Build the Sim executable" ON)
if(BUILD_SIM_APP)
  add_executable(sim src/main.cpp)
  target_link_libraries(sim PRIVATE simcore)
  # Place the binary under build/Sim/
  set_target_properties(sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Sim")
endif()
