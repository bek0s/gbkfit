
set(TARGET_NAME libgbkfit_openmp)

project(${TARGET_NAME}-project CXX)

set(FFTW3_USE_STATIC_LIBS 1)

find_package(fftw3)
find_package(OpenMP)
find_package(pybind11)

set(HEADER_FILES
  include/gbkfit/openmp/dmodels.hpp
  include/gbkfit/openmp/gmodels.hpp
  include/gbkfit/openmp/kernels_common.hpp
  include/gbkfit/openmp/kernels_main.hpp
  include/gbkfit/openmp/kernels_math.hpp
  include/gbkfit/openmp/kernels_misc.hpp
  include/gbkfit/openmp/kernels_traits.hpp

  include/gbkfit/openmp/fftw3.hpp
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
  OpenMP::OpenMP_CXX
  fftw3::fftw3
  fftw3::fftw3_omp
  fftw3::fftw3f
  fftw3::fftw3f_omp
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
