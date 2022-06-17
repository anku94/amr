cmake_minimum_required (VERSION 3.1)
project(tau-stuff C CXX)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(MPI MODULE REQUIRED)
find_package(TAU MODULE REQUIRED)

add_executable(test test.c)
target_link_libraries(test PRIVATE MPI::MPI_C TAU::TAU)
