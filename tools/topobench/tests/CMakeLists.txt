find_package(GTest CONFIG)

if (NOT GTest_FOUND)
  message(WARNING "GTest not found, tests will not be built")
  return()
endif()

set(amr-tests topogen_test.cc)

set(test_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(topogen_test topogen_test.cc)
add_test(NAME topogen_test COMMAND topogen_test)
target_link_libraries(topogen_test GTest::gtest_main topo)
target_include_directories(topogen_test PRIVATE ${test_include_dirs})
gtest_discover_tests(topogen_test)
