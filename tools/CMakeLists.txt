cmake_minimum_required(VERSION 3.21)
project(amr-tools)

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

set (common_include "${CMAKE_SOURCE_DIR}/common/")
set (common_srcs "${CMAKE_SOURCE_DIR}/common/common.cc")

# BEGIN GTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
# END GTest

option(FORT_ENABLE_TESTING OFF)
option(FORT_TEST_BUILD OFF)

add_subdirectory(external/libfort)
add_subdirectory(topobench)
add_subdirectory(loadbalancing)
