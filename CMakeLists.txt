cmake_minimum_required(VERSION 3.20)
project(SpaceforgeSim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Tell CMake where SPARTA (headers + lib) lives
if(NOT DEFINED SPARTA_DIR)
  if(DEFINED ENV{SPARTA_DIR})
    set(SPARTA_DIR $ENV{SPARTA_DIR})
  else()
    message(FATAL_ERROR "Provide -DSPARTA_DIR=/full/path/to/sparta/src")
  endif()
endif()

add_executable(simulation src/main.cpp)

# Headers are in sparta/src
target_include_directories(simulation PRIVATE ${SPARTA_DIR})

# Import the prebuilt static library
add_library(sparta_static STATIC IMPORTED GLOBAL)
if(EXISTS "${SPARTA_DIR}/libsparta.a")
  set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta.a")
elseif(EXISTS "${SPARTA_DIR}/libsparta_mpi.a")
  set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta_mpi.a")
else()
  message(FATAL_ERROR "Couldn't find libsparta.a or libsparta_mpi.a under ${SPARTA_DIR}")
endif()

target_link_libraries(simulation PRIVATE sparta_static MPI::MPI_CXX)
