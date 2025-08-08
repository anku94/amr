find_package(GTest CONFIG)

if(NOT GTest_FOUND)
  message(STATUS "GTest not found, skipping tests")
  return()
endif()

set(lb_tests lb_tests.cc lb_util_tests.cc lb_chunkwise_tests.cc policy_tests.cc)

foreach(lcv ${lb_tests})
  get_filename_component(id ${lcv} NAME_WE)
  message(STATUS "Building test: ${id}")
  add_executable(${id} ${lcv})
  target_link_libraries(${id} GTest::gtest_main lb tools-common)
  target_include_directories(${id} PRIVATE ${all_include_dirs})
  gtest_discover_tests(${id})
  add_test(NAME ${id} COMMAND ${id})
endforeach()
