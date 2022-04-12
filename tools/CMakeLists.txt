cmake_minimum_required(VERSION 3.21)
project(tools)

set(CMAKE_CXX_STANDARD 14)

# setup cache variables for ccmake users
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
            CACHE STRING "Choose the type of build" FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (DEBUG_SANITIZER Off CACHE STRING "Sanitizer for debug builds")
set_property (CACHE DEBUG_SANITIZER PROPERTY STRINGS
        "Off" "Address" "Thread")
set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")

set (as_flags "-fsanitize=address -O1 -fno-omit-frame-pointer")
set (ts_flags "-fsanitize=thread  -O1 -fno-omit-frame-pointer")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    if (${DEBUG_SANITIZER} STREQUAL "Address")
        set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${as_flags}")
        set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${as_flags}")
    elseif (${DEBUG_SANITIZER} STREQUAL "Thread")
        set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${ts_flags}")
        set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ts_flags}")
    endif ()
endif ()

add_executable(topobench topobench.cc block.cc driver.cc logger.cc topology.cc common.cc logger.h logger.cc)

# Find required packages
find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

#
# make sure we link with MPI.  use "MPI_CXX_COMPILE_FLAGS_LIST"
# prepared by the calling module.
#
foreach (tgt topobench)

    # mpich on ub14 gives a leading space that we need to trim off
    foreach (lcv ${MPI_CXX_COMPILE_FLAGS_LIST})
        if (NOT ${lcv} STREQUAL "")
            target_compile_options (${tgt}
                    PUBLIC $<BUILD_INTERFACE:${lcv}>)
        endif ()
    endforeach ()

    # XXX: have to do this one dir at a time
    foreach (lcv ${MPI_CXX_INCLUDE_PATH})
        target_include_directories (${tgt}
                PUBLIC $<BUILD_INTERFACE:${lcv}>)
    endforeach ()

    foreach (lcv ${MPI_CXX_LIBRARIES})
        target_link_libraries(${tgt} $<BUILD_INTERFACE:${lcv}>)
    endforeach ()

    set_property (TARGET ${tgt} APPEND PROPERTY LINK_FLAGS
            ${MPI_CXX_LINK_FLAGS})

endforeach ()
