
set(TARGET_NAME libgbkfit)

project(${TARGET_NAME}-project CXX)

add_library(
  ${TARGET_NAME}
  INTERFACE
)

target_include_directories(
  ${TARGET_NAME}
  INTERFACE
  include
)
