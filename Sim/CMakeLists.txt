cmake_minimum_required(VERSION 3.20)
project(SimCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all Sim .cpp files EXCEPT Sim/src/main.cpp (that’s the app entry)
file(GLOB SIM_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SIM_SRC EXCLUDE REGEX ".*/main\\.cpp$")

# Build the reusable library
add_library(simcore ${SIM_SRC})
target_include_directories(simcore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link SPARTA + MPI here so everything that uses simcore is covered
target_link_libraries(simcore PUBLIC sparta_static MPI::MPI_CXX)

# Give code a stable way to find input/ at runtime
target_compile_definitions(simcore PUBLIC PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Optional: build Sim’s own app from Sim/src/main.cpp
option(BUILD_SIM_APP "Build Sim/src/main.cpp" ON)
if(BUILD_SIM_APP)
  add_executable(sim_app ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
  target_link_libraries(sim_app PRIVATE simcore)
  target_include_directories(sim_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
