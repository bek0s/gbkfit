
set(TARGET_NAME libgbkfit_cuda)

project(${TARGET_NAME}-project CUDA CXX)

find_package(pybind11)
find_package(CUDA REQUIRED)

set(HEADER_FILES
  include/gbkfit/cuda/common.hpp
  include/gbkfit/cuda/dmodels.hpp
  include/gbkfit/cuda/fftutils.hpp
  include/gbkfit/cuda/gmodels.hpp
  include/gbkfit/cuda/kernels.hpp
  include/gbkfit/cuda/wrapper.hpp
)

set(SOURCE_FILES
  src/bindings.cpp
  src/dmodels.cpp
  src/gmodels.cpp
  src/wrapper.cu
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

pybind11_add_module(
  ${TARGET_NAME}
  NO_EXTRAS
  ${HEADER_FILES}
  ${SOURCE_FILES}
)

target_include_directories(
  ${TARGET_NAME}
  PRIVATE
  include
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
  ${TARGET_NAME}
  PRIVATE
  libgbkfit
  ${CUDA_CUFFT_LIBRARIES}
)

target_compile_options(
  ${TARGET_NAME}
  PRIVATE
)

target_compile_definitions(
  ${TARGET_NAME}
  PRIVATE
  EXTENSION_NAME=${TARGET_NAME}
)

set_target_properties(
  ${TARGET_NAME}
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
)
