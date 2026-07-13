# Native analysis core. Two consumers share this file:
#   1. scikit-build-core (pip install / just build) — defines SKBUILD and only
#      needs the _native Python module.
#   2. Standalone test builds (just test-cpp) — configure with
#      -DF1LAB_BUILD_TESTS=ON to compile the Catch2 suite, no Python required.
cmake_minimum_required(VERSION 3.26)
project(f1lab_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The pure-C++ core: no Python, no pybind11. Kept as a separate static
# library so the unit tests can link it without building the bindings.
add_library(f1lab_core STATIC src/delta.cpp src/corners.cpp)
target_include_directories(f1lab_core PUBLIC src)
set_target_properties(f1lab_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(f1lab_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/bindings.cpp)
  target_link_libraries(_native PRIVATE f1lab_core)
  # Lands next to the pure-Python sources inside the f1lab package.
  install(TARGETS _native LIBRARY DESTINATION f1lab)
endif()

option(F1LAB_BUILD_TESTS "Build the Catch2 unit tests" OFF)
if(F1LAB_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(f1lab_tests tests/test_delta.cpp tests/test_corners.cpp)
  target_link_libraries(f1lab_tests PRIVATE f1lab_core Catch2::Catch2WithMain)

  include(CTest)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  catch_discover_tests(f1lab_tests)
endif()
