include(GoogleTest)

set(amr-tests topogen_test.cc)

foreach (lcv ${amr-tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} EXCLUDE_FROM_ALL ${lcv})
    target_link_libraries(${id} GTest::gtest_main topo)
    gtest_discover_tests(${id})
endforeach()