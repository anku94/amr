project(amr-tools)
cmake_minimum_required(VERSION 3.10)

set (common_include "${CMAKE_CURRENT_SOURCE_DIR}/common/")
set (common_srcs "${CMAKE_CURRENT_SOURCE_DIR}/common/common.cc")

add_subdirectory(topobench)
add_subdirectory(loadbalancing)
add_subdirectory(ember-benchmarks)
