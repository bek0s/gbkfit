
# The kernels need OpenMP 3.x; with MSVC, that means /openmp:llvm
set(OpenMP_RUNTIME_MSVC llvm)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

# NOMINSIZE: nanobind optimises for size by default; we want speed
nanobind_add_module(
  _host
  STABLE_ABI
  NB_STATIC
  NOMINSIZE
  LTO
  src/bindings.cpp
)

target_include_directories(
  _host
  PRIVATE
  include
  ${PROJECT_SOURCE_DIR}/src/native/bindings/include
  ${PROJECT_SOURCE_DIR}/src/native/third_party/pocketfft
)

target_link_libraries(
  _host
  PRIVATE
  gbkfit::core
  OpenMP::OpenMP_CXX
  Threads::Threads
)

target_compile_definitions(
  _host
  PRIVATE
  EXTENSION_NAME=_host
)

install(TARGETS _host LIBRARY DESTINATION ${GBKFIT_NATIVE_INSTALL_DIR})
