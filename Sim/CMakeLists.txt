cmake_minimum_required(VERSION 3.15)
project(SpaceForgeSimCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source and include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SIM_SOURCES src/*.cpp)

# Build the simulation executable
add_executable(sim ${SIM_SOURCES})