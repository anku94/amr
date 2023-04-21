## Status: incomplete. Complex dependencies.
## Check pkgconfig for /usr/local/opt/cbc for dependency tree.

if (APPLE)
    set(CBC_SEARCH_PATH /usr/local/opt)
endif ()

set(CBC_LIBS
        CBC::Cbc
        CBC::Clp
        CBC::CoinUtils)

find_path(CBC_INCLUDE_DIR
        NAMES cbc/coin/CbcModel.hpp
        HINTS ${CBC_SEARCH_PATH}
        PATH_SUFFIXES cbc/include)

find_library(CBC_LIBRARY
        NAMES CbcSolver
        HINTS ${CBC_SEARCH_PATH}
        PATH_SUFFIXES cbc/lib)

find_path(CLP_INCLUDE_DIR
        NAMES clp/coin/ClpModel.hpp
        HINTS ${CBC_SEARCH_PATH}
        PATH_SUFFIXES clp/include)

find_library(CLP_LIBRARY
        NAMES CbcSolver
        HINTS ${CBC_SEARCH_PATH}
        PATH_SUFFIXES cbc/lib)

message(STATUS "FindCBC Running: ${CBC_INCLUDE_DIR} ${CBC_LIBRARY}")
message(STATUS "FindCLP Running: ${CLP_INCLUDE_DIR} ${CLP_LIBRARY}")
# set(CBC_INCLUDE_DIR "${CBC_INCLUDE_DIRS}/cbc")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBC DEFAULT_MSG CBC_INCLUDE_DIRS CBC_LIBRARY)

add_library(CBC::Cbc UNKNOWN IMPORTED)
set_target_properties(CBC::Cbc PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CBC_INCLUDE_DIRS}/cbc
        IMPORTED_LOCATION ${CBC_LIBRARY})

add_library(CBC::Clp UNKNOWN IMPORTED)
set_target_properties(CBC::Clp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CLP_INCLUDE_DIRS}/clp
        IMPORTED_LOCATION ${CLP_LIBRARY})
