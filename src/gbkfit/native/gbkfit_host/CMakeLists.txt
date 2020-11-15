
set(TARGET_NAME libgbkfit_host)

project(${TARGET_NAME}-project CXX)

find_package(fftw3)
find_package(OpenMP)

set(HEADER_FILES
  include/gbkfit/host/common.hpp
  include/gbkfit/host/dmodels.hpp
  include/gbkfit/host/fftutils.hpp
  include/gbkfit/host/gmodels.hpp
  include/gbkfit/host/kernels.hpp
)

set(SOURCE_FILES
  src/bindings.cpp
  src/dmodels.cpp
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
  fftw3::fftw3
  fftw3::fftw3_threads
  fftw3::fftw3f
  fftw3::fftw3f_threads
  OpenMP::OpenMP_CXX
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
