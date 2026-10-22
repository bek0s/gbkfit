
# The DNest4 sampler (third_party/dnest4), and the module that runs it on
# models whose likelihood is a Python function

set(DNEST4_DIR ${PROJECT_SOURCE_DIR}/src/native/third_party/dnest4)

add_library(
  dnest4
  STATIC
  ${DNEST4_DIR}/Barrier.cpp
  ${DNEST4_DIR}/Level.cpp
  ${DNEST4_DIR}/LikelihoodType.cpp
  ${DNEST4_DIR}/Options.cpp
  ${DNEST4_DIR}/RNG.cpp
  ${DNEST4_DIR}/Utils.cpp
  ${DNEST4_DIR}/Distributions/Cauchy.cpp
  ${DNEST4_DIR}/Distributions/ContinuousDistribution.cpp
  ${DNEST4_DIR}/Distributions/Exponential.cpp
  ${DNEST4_DIR}/Distributions/Gaussian.cpp
  ${DNEST4_DIR}/Distributions/Jeffreys.cpp
  ${DNEST4_DIR}/Distributions/Kumaraswamy.cpp
  ${DNEST4_DIR}/Distributions/Laplace.cpp
  ${DNEST4_DIR}/Distributions/LogUniform.cpp
  ${DNEST4_DIR}/Distributions/Pareto.cpp
  ${DNEST4_DIR}/Distributions/Rayleigh.cpp
  ${DNEST4_DIR}/Distributions/Triangular.cpp
  ${DNEST4_DIR}/Distributions/Uniform.cpp
  ${DNEST4_DIR}/RJObject/ConditionalPriors/BasicCircular.cpp
  ${DNEST4_DIR}/RJObject/ConditionalPriors/ClassicMassInf.cpp
  ${DNEST4_DIR}/RJObject/ConditionalPriors/ClassicMassInf1D.cpp
  ${DNEST4_DIR}/RJObject/ConditionalPriors/ConditionalPrior.cpp
)

set_target_properties(dnest4 PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(dnest4 PUBLIC ${DNEST4_DIR})

target_compile_features(dnest4 PUBLIC cxx_std_17)

# One thread (the likelihood holds the GIL), and no state file in the
# working directory
target_compile_definitions(
  dnest4
  PUBLIC
  NO_THREADS
  DNEST4_NO_SAMPLER_STATE
)

find_package(Threads REQUIRED)

# NOMINSIZE: nanobind optimises for size by default; we want speed
nanobind_add_module(
  _dnest4
  STABLE_ABI
  NB_STATIC
  NOMINSIZE
  LTO
  src/bindings.cpp
)

target_link_libraries(
  _dnest4
  PRIVATE
  dnest4
  Threads::Threads
)

target_compile_definitions(
  _dnest4
  PRIVATE
  EXTENSION_NAME=_dnest4
)

install(TARGETS _dnest4 LIBRARY DESTINATION gbkfit/fitting/fitters/dnest4)
