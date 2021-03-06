# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Find fftw3 libraries and include directories.
#
# http://www.fftw.org/
#
# Sets the following variables:
#
#   fftw3_FOUND
#   fftw3_INCLUDE_DIR
#   fftw3_INCLUDE_DIRS
#   fftw3_DOUBLE_LIBRARY
#   fftw3_DOUBLE_OMP_LIBRARY
#   fftw3_DOUBLE_THREADS_LIBRARY
#   fftw3_SINGLE_LIBRARY
#   fftw3_SINGLE_OMP_LIBRARY
#   fftw3_SINGLE_THREADS_LIBRARY
#   fftw3_LONGDOUBLE_LIBRARY
#   fftw3_LONGDOUBLE_OMP_LIBRARY
#   fftw3_LONGDOUBLE_THREADS_LIBRARY
#   fftw3_LIBRARIES
#
# Checks the following variables:
#
#   fftw3_USE_STATIC_LIBS
#
# Checks the following environment variables:
#
#   FFTW3_ROOT
#
# Declares the following targets:
#
#   fftw3::all
#   fftw3::fftw3
#   fftw3::fftw3_omp
#   fftw3::fftw3_threads
#   fftw3::fftw3f
#   fftw3::fftw3f_omp
#   fftw3::fftw3f_threads
#   fftw3::fftw3l
#   fftw3::fftw3l_omp
#   fftw3::fftw3l_threads
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

  set(INCLUDE_SEARCH_PATHS
    "$ENV{FFTW3_ROOT}/include"
  )
  set(INCLUDE_SEARCH_HINTS
  )
  set(LIBRARY_SEARCH_PATHS
    "$ENV{FFTW3_ROOT}/lib"
  )
  set(LIBRARY_SEARCH_HINTS
  )

elseif(UNIX)

  set(INCLUDE_SEARCH_PATHS
    "$ENV{FFTW3_ROOT}/include"
    "/usr/include"
    "/usr/fftw3/include"
    "/usr/local/include"
    "/usr/local/fftw3/include"
  )
  set(INCLUDE_SEARCH_HINTS
  )
  set(LIBRARY_SEARCH_PATHS
    "$ENV{FFTW3_ROOT}/lib"
    "/usr/lib"
    "/usr/fftw3/lib"
    "/usr/local/lib"
    "/usr/local/fftw3/lib"
  )
  set(LIBRARY_SEARCH_HINTS
  )

else()

  message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(fftw3_INCLUDE_DIR
  NAMES
  "fftw3.h"
  HINTS
  ${INCLUDE_SEARCH_HINTS}
  PATHS
  ${INCLUDE_SEARCH_PATHS}
  DOC "Absolute path to fftw3 include directory.")

#
# Set library names. These names will also be used for the export targets.
#

set(LIBRARY_LIB_NAME_LIST
  "fftw3"
  "fftw3_omp"
  "fftw3_threads"
  "fftw3f"
  "fftw3f_omp"
  "fftw3f_threads"
  "fftw3l"
  "fftw3l_omp"
  "fftw3l_threads"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
  "fftw3_DOUBLE_LIBRARY"
  "fftw3_DOUBLE_OMP_LIBRARY"
  "fftw3_DOUBLE_THREADS_LIBRARY"
  "fftw3_SINGLE_LIBRARY"
  "fftw3_SINGLE_OMP_LIBRARY"
  "fftw3_SINGLE_THREADS_LIBRARY"
  "fftw3_LONGDOUBLE_LIBRARY"
  "fftw3_LONGDOUBLE_OMP_LIBRARY"
  "fftw3_LONGDOUBLE_THREADS_LIBRARY"
)

# Set library suffixes
if(${fftw3_USE_STATIC_LIBS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES_DEFAULT ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

#
# Detect the paths of the above libraries and declare export targets.
#

list(LENGTH LIBRARY_LIB_VARIABLE_NAME_LIST LIBRARY_LIB_NAME_LIST_LENGTH)
MATH(EXPR LIBRARY_LIB_NAME_LIST_LENGTH "${LIBRARY_LIB_NAME_LIST_LENGTH}-1")
foreach(i RANGE ${LIBRARY_LIB_NAME_LIST_LENGTH})
  list(GET LIBRARY_LIB_VARIABLE_NAME_LIST ${i} LIB_NAME_VAR)
  list(GET LIBRARY_LIB_NAME_LIST ${i} LIB_NAME)
  find_library(${LIB_NAME_VAR}
    NAMES
    ${LIB_NAME}
    HINTS
    ${LIBRARY_SEARCH_HINTS}
    PATHS
    ${LIBRARY_SEARCH_PATHS}
    DOC "Absolute path to ${LIB_NAME} library."
  )
  add_library(fftw3::${LIB_NAME} INTERFACE IMPORTED)
  set_target_properties(fftw3::${LIB_NAME}
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${fftw3_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${${LIB_NAME_VAR}}"
  )
endforeach(i)

# Restore default library suffixes
if(${fftw3_USE_STATIC_LIBS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_DEFAULT})
endif()

#
# Combine all library paths into one variable.
#

set(fftw3_INCLUDE_DIRS
  ${fftw3_INCLUDE_DIR}
)

set(fftw3_LIBRARIES
  ${fftw3_DOUBLE_LIBRARY}
  ${fftw3_DOUBLE_OMP_LIBRARY}
  ${fftw3_DOUBLE_THREADS_LIBRARY}
  ${fftw3_SINGLE_LIBRARY}
  ${fftw3_SINGLE_OMP_LIBRARY}
  ${fftw3_SINGLE_THREADS_LIBRARY}
  ${fftw3_LONGDOUBLE_LIBRARY}
  ${fftw3_LONGDOUBLE_OMP_LIBRARY}
  ${fftw3_LONGDOUBLE_THREADS_LIBRARY}
)

add_library(fftw3::all INTERFACE IMPORTED)
set_target_properties(fftw3::all
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${fftw3_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${fftw3_LIBRARIES}"
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(fftw3
  DEFAULT_MSG
  fftw3_LIBRARIES fftw3_INCLUDE_DIRS)

#
# Unset all the temporary variables. Do I need to do this?
#

unset(LIB_NAME)
unset(LIB_NAME_VAR)
unset(LIBRARY_LIB_VARIABLE_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST_LENGTH)
unset(INCLUDE_SEARCH_PATHS)
unset(INCLUDE_SEARCH_HINTS)
unset(LIBRARY_SEARCH_PATHS)
unset(LIBRARY_SEARCH_HINTS)
