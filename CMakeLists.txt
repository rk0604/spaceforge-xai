cmake_minimum_required(VERSION 3.20)
project(SpaceForge LANGUAGES CXX)

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
set(SPARTA_DIR ${SPARTA_DIR} CACHE PATH "SPARTA src dir (headers + lib)")

# Import the prebuilt static SPARTA library
add_library(sparta_static STATIC IMPORTED GLOBAL)
if(EXISTS "${SPARTA_DIR}/libsparta_mpi.a")
  set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta_mpi.a")
elseif(EXISTS "${SPARTA_DIR}/libsparta.a")
  set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta.a")
else()
  message(FATAL_ERROR "Couldn't find libsparta*.a under ${SPARTA_DIR}")
endif()

# Make SPARTA headers visible to anything that links sparta_static
set_property(TARGET sparta_static PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SPARTA_DIR}")

# Build Sim (library + optional app)
add_subdirectory(Sim)

# (Optional) keep your old standalone SPARTA runner if you still want it:
# add_executable(sparta_trial src/main.cpp)
# target_link_libraries(sparta_trial PRIVATE sparta_static MPI::MPI_CXX simcore)
# target_compile_definitions(sparta_trial PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
