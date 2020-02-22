
cmake_minimum_required(VERSION 3.10)

project(libgbkfit-workspace)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/pybind11")
  add_subdirectory(third_party/pybind11)
endif()

add_subdirectory(gbkfit)

if($ENV{GBKFIT_BUILD_HOST})
  add_subdirectory(gbkfit_host)
endif()

if($ENV{GBKFIT_BUILD_CUDA})
  add_subdirectory(gbkfit_cuda)
endif()
