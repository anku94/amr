cmake_minimum_required (VERSION 3.1)
project (amr-tau-plugin C CXX)
set(CMAKE_CXX_STANDARD 14)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(TAU MODULE)

if (NOT TAU_FOUND)
  message(STATUS "TAU not found. Not building AMR TAU plugin.")
  return()
endif()

set(PLUGIN_SOURCES Tau_plugin_amr.cc amr_tracer.cc amr_util.cc ../tools/common/common.cc)
add_library(TAU-amr SHARED ${PLUGIN_SOURCES})
target_link_libraries(TAU-amr TAU::TAU)

install(TARGETS TAU-amr
  LIBRARY DESTINATION lib)

# reusing this path, hacky
install(FILES cmake/FindTAU.cmake
  DESTINATION share/cmake/pdlfs-common)
