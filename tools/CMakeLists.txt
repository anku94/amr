project(amr-tools)
cmake_minimum_required(VERSION 3.10)

set (common_include "${CMAKE_CURRENT_SOURCE_DIR}/common/")
set (common_srcs "${CMAKE_CURRENT_SOURCE_DIR}/common/common.cc")

option(FORT_ENABLE_TESTING OFF)
option(FORT_TEST_BUILD OFF)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(external/libfort)
set_target_properties(fort PROPERTIES EXCLUDE_FROM_ALL TRUE)
set(BUILD_SHARED_LIBS ON)

add_subdirectory(topobench)
add_subdirectory(loadbalancing)
add_subdirectory(ember-benchmarks)
