
# Header-only code shared by the host and cuda modules

add_library(gbkfit_core INTERFACE)
add_library(gbkfit::core ALIAS gbkfit_core)

target_include_directories(gbkfit_core INTERFACE include)

target_compile_features(gbkfit_core INTERFACE cxx_std_20)
