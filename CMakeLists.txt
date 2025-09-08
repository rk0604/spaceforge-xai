cmake_minimum_required(VERSION 3.20)
project(SpaceForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

# ---- SPARTA toggle (OFF lets you build/run without SPARTA) ----
option(ENABLE_SPARTA "Enable SPARTA coupling" OFF)

if(ENABLE_SPARTA)
  if(NOT DEFINED SPARTA_DIR)
    if(DEFINED ENV{SPARTA_DIR})
      set(SPARTA_DIR $ENV{SPARTA_DIR})
    else()
      message(FATAL_ERROR "Provide -DSPARTA_DIR=/full/path/to/sparta OR -DENABLE_SPARTA=OFF")
    endif()
  endif()

  add_library(sparta_static STATIC IMPORTED GLOBAL)
  if(EXISTS "${SPARTA_DIR}/libsparta_mpi.a")
    set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta_mpi.a")
  elseif(EXISTS "${SPARTA_DIR}/libsparta.a")
    set_target_properties(sparta_static PROPERTIES IMPORTED_LOCATION "${SPARTA_DIR}/libsparta.a")
  else()
    message(FATAL_ERROR "Couldn't find libsparta*.a under ${SPARTA_DIR}")
  endif()
  set_property(TARGET sparta_static PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SPARTA_DIR}")
else()
  # Stub so downstream targets can link even without SPARTA installed
  add_library(sparta_static INTERFACE)
endif()

# Build Sim subtree
add_subdirectory(Sim)
