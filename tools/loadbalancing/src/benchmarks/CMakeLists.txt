find_package(GTest CONFIG)

if (NOT GTest_FOUND)
  return()
endif()

set(benchmark_srcs benchmark.cc
    ../distrib/alias_method.cc
    ../common/globals.cc
    ../common/config_parser.cc)
add_executable(benchmark benchmark_main.cc)
target_link_libraries(benchmark PRIVATE lb lb_common)
target_include_directories(benchmark 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(benchmark-test benchmark_test.cc)
target_link_libraries(benchmark-test PRIVATE lb lb_common GTest::gtest GTest::gtest_main)
target_include_directories(benchmark-test 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)
gtest_discover_tests(benchmark-test)
