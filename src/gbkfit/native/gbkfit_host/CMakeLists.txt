
set(TARGET_NAME libgbkfit_host)

project(${TARGET_NAME}-project CXX)

set(FFTW3_USE_STATIC_LIBS 1)

find_package(fftw3)
find_package(OpenMP)
find_package(pybind11)

set(HEADER_FILES
  include/gbkfit/drivers/host/common.hpp
  include/gbkfit/drivers/host/dmodels.hpp
  include/gbkfit/drivers/host/fftw3.hpp
  include/gbkfit/drivers/host/gmodels.hpp
  include/gbkfit/drivers/host/kernels_common.hpp
  include/gbkfit/drivers/host/kernels_main.hpp
  include/gbkfit/drivers/host/kernels_math.hpp
  include/gbkfit/drivers/host/kernels_misc.hpp
  include/gbkfit/drivers/host/kernels_traits.hpp
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
  fftw3::fftw3_omp
  fftw3::fftw3f
  fftw3::fftw3f_omp
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
