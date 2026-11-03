
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
# Ask Python where nanobind is, so that builds outside pip find it too
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(src/native/core)

if(GBKFIT_HOST)
  add_subdirectory(src/native/host)
endif()

if(GBKFIT_CUDA)
  add_subdirectory(src/native/cuda)
endif()

# The bridge between Python and DNest4, always built.
# The code of DNest4 itself is in src/native/third_party/dnest4.
add_subdirectory(src/native/dnest4)
