
find_package(OpenMP REQUIRED COMPONENTS CXX)

# FFTW3 (double and single precision, plus their threads libraries).
# Set FFTW3_ROOT or CMAKE_PREFIX_PATH if it is not in a system location.
find_path(FFTW3_INCLUDE_DIR fftw3.h
  REQUIRED HINTS ENV FFTW3_ROOT PATH_SUFFIXES include)
set(FFTW3_LIBRARIES)
foreach(name fftw3f_threads fftw3f fftw3_threads fftw3)
  find_library(FFTW3_${name}_LIBRARY ${name}
    REQUIRED HINTS ENV FFTW3_ROOT PATH_SUFFIXES lib lib64)
  list(APPEND FFTW3_LIBRARIES ${FFTW3_${name}_LIBRARY})
endforeach()

pybind11_add_module(
  _host
  src/bindings.cpp
  src/dmodels.cpp
  src/gmodels.cpp
  src/objective.cpp
)

target_include_directories(
  _host
  PRIVATE
  include
  ${FFTW3_INCLUDE_DIR}
)

target_link_libraries(
  _host
  PRIVATE
  gbkfit::core
  ${FFTW3_LIBRARIES}
  OpenMP::OpenMP_CXX
)

target_compile_definitions(
  _host
  PRIVATE
  EXTENSION_NAME=_host
)

install(TARGETS _host LIBRARY DESTINATION ${GBKFIT_NATIVE_INSTALL_DIR})
