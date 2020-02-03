
set(TARGET_NAME libgbkfit_cuda)

project(${TARGET_NAME}-project CXX CUDA)

find_package(pybind11)

set(HEADER_FILES
  include/gbkfit/cuda/dmodels.hpp
  include/gbkfit/cuda/gmodels.hpp
  include/gbkfit/cuda/kernels_main.hpp
  include/gbkfit/cuda/kernels_math.hpp
  include/gbkfit/cuda/kernels_traits.hpp
  include/gbkfit/cuda/wrapper.hpp
)

set(SOURCE_FILES
  src/bindings.cpp
  src/dmodels.cpp
  src/gmodels.cpp
  src/wrapper.cu
)

pybind11_add_module(
  ${TARGET_NAME}
  ${HEADER_FILES}
  ${SOURCE_FILES}
)

target_include_directories(
  ${TARGET_NAME}
  PRIVATE
  include
)

target_link_libraries(
  ${TARGET_NAME}
  PRIVATE
  libgbkfit
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
