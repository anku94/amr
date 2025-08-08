find_package(GTest CONFIG)

if (NOT GTest_FOUND)
  return()
endif()

set(benchmark_srcs benchmark.cc)
add_executable(policybench benchmark_main.cc)
target_link_libraries(policybench PRIVATE lb tools-common)
target_include_directories(policybench 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(policybench-test benchmark_test.cc)
target_link_libraries(policybench-test PRIVATE lb tools-common GTest::gtest_main)
target_include_directories(policybench-test 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)
gtest_discover_tests(policybench-test)