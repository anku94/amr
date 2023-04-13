include(GoogleTest)

set(lb_tests lb_tests.cc misc_tests.cc)

foreach (lcv ${lb_tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} EXCLUDE_FROM_ALL ${lcv})
    target_link_libraries(${id} GTest::gtest_main lb)
    gtest_discover_tests(${id})
endforeach()
