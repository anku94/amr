find_package(GTest CONFIG)

if (NOT GTest_FOUND)
  return()
endif()

set(benchmark_srcs benchmark.cc alias_method.cc ../globals.cc ../config_parser.cc)
add_executable(benchmark benchmark_main.cc ${benchmark_srcs})
target_link_libraries(benchmark PRIVATE lb pdlfs-common)

add_executable(benchmark-test benchmark_test.cc ${benchmark_srcs})
target_link_libraries(benchmark-test PRIVATE lb pdlfs-common GTest::gtest GTest::gtest_main)
gtest_discover_tests(benchmark-test)
