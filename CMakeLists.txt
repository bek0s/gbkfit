
cmake_minimum_required(VERSION 3.26...4.4)

project(gbkfit LANGUAGES CXX)

option(GBKFIT_HOST "Build the host (CPU/OpenMP) native module" ON)
option(GBKFIT_CUDA "Build the CUDA native module" OFF)

if(NOT GBKFIT_HOST AND NOT GBKFIT_CUDA)
  message(FATAL_ERROR
    "No native module selected; enable GBKFIT_HOST and/or GBKFIT_CUDA")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

if(GBKFIT_CUDA)
  # Default to the GPU(s) present on the build machine.
  # Override with -DCMAKE_CUDA_ARCHITECTURES=... for redistributable builds.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
endif()

find_package(Python 3.12 REQUIRED
  COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Python package directory the native modules are installed into
set(GBKFIT_NATIVE_INSTALL_DIR gbkfit/driver/native)

add_subdirectory(src/native/core)

if(GBKFIT_HOST)
  add_subdirectory(src/native/host)
endif()

if(GBKFIT_CUDA)
  add_subdirectory(src/native/cuda)
endif()
