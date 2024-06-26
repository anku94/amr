cmake_minimum_required(VERSION 3.16)
project(amr-tools)

# To be able to use TAU_ROOT
cmake_policy(SET CMP0074 NEW)

find_package(GTest)
if (NOT GTEST_FOUND)
  message(STATUS "GoogleTest not found: skipping tests")
  set (BUILD_TESTING OFF)
else()
  include(GoogleTest)
  include(CTest)
endif()

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
set (BUILD_TAU_PLUGIN Off CACHE BOOL "Build TAU plugin")

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

add_subdirectory(decks)
add_subdirectory(plugins)
add_subdirectory(scripts)
add_subdirectory(tools)
