
find_package(CUDAToolkit REQUIRED)

# NOMINSIZE: nanobind optimises for size by default; we want speed
nanobind_add_module(
  _cuda
  STABLE_ABI
  NB_STATIC
  NOMINSIZE
  src/bindings.cpp
  src/gmodels.cpp
  src/wrapper.cu
)

target_include_directories(
  _cuda
  PRIVATE
  include
  ${PROJECT_SOURCE_DIR}/src/native/bindings/include
)

# Link the CUDA runtime statically, so that the module only needs the
# NVIDIA driver at run time and can be shipped inside a wheel
target_link_libraries(
  _cuda
  PRIVATE
  gbkfit::core
  CUDA::cudart_static
)

target_compile_features(
  _cuda
  PRIVATE
  cuda_std_20
)

target_compile_options(
  _cuda
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_definitions(
  _cuda
  PRIVATE
  EXTENSION_NAME=_cuda
)

install(TARGETS _cuda LIBRARY DESTINATION ${GBKFIT_NATIVE_INSTALL_DIR})
