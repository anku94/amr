cmake_minimum_required (VERSION 3.1)
project (amr-tau-plugin C CXX)
set(CMAKE_CXX_STANDARD 14)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(TAU REQUIRED)

if (TAU_FOUND)
  set(PLUGIN_SOURCES Tau_plugin_amr.cc)
  add_library(TAU-amr ${PLUGIN_SOURCES})
  target_link_libraries(TAU-amr TAU)
else ()
  message(FATAL_ERROR "TAU not found")
endif()
