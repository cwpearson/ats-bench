# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.214.tar.gz"
    SHA1 "e14bc153a7f16d6a5eeec845fb0283c8fad8c358"
)

project(ats-bench LANGUAGES CUDA VERSION 0.1.0)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

#https://blog.kitware.com/cmake-and-the-default-build-type/
# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

hunter_add_package(spdlog)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting verbose build during Debug")
  set(CMAKE_VERBOSE_MAKEFILE ON)
#   add_definitions(-DSPDLOG_TRACE_ON)
#   add_definitions(-DSPDLOG_DEBUG_ON)
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  add_definitions(-DNDEBUG)
#   add_definitions(-DSPDLOG_DEBUG_ON)
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()

# removed -Wredundant-decls for cuda 10.1
# removed -Wundef for cuda 10.0
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
 -Xcompiler=-Wall\
 -Xcompiler=-Wextra\
 -Xcompiler=-Wcast-qual \
 -Xcompiler=-Wcast-align \
 -Xcompiler=-Wstrict-aliasing \
 -Xcompiler=-Wpointer-arith \
 -Xcompiler=-Winit-self \
 -Xcompiler=-Wshadow \
 -Xcompiler=-Wswitch-enum \
 -Xcompiler=-Wfloat-equal \
 -Xcompiler=-Wvla\
 -Xcompiler=-fmax-errors=1 \
 -Xcompiler=-Wfatal-errors\
 ")

 set(CMAKE_CUDA_STANDARD 11)

include_directories(SYSTEM thirdparty)
include_directories(src)

find_package(spdlog CONFIG REQUIRED)

add_subdirectory(src)