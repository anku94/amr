if (NOT BUILD_TESTING)
  return()
endif()

find_package(GTest CONFIG)

set(benchmark_srcs benchmark.cc alias_method.cc)
add_executable(benchmark benchmark_main.cc ${benchmark_srcs})
target_link_libraries(benchmark PRIVATE lb pdlfs-common)

add_executable(benchmark-test benchmark_test.cc ${benchmark_srcs})
target_link_libraries(benchmark-test PRIVATE lb pdlfs-common GTest::gtest GTest::gtest_main)
gtest_discover_tests(benchmark-test)
