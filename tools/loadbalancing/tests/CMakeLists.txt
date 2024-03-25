if (NOT BUILD_TESTING)
  return()
endif()

find_package(GTest CONFIG)
find_package(pdlfs-common CONFIG REQUIRED)

set(lb_tests lb_tests.cc misc_tests.cc policy_tests.cc scale_tests.cc)


foreach (lcv ${lb_tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} ${lcv})
    target_link_libraries(${id} GTest::gtest_main lb pdlfs-common fort)
    target_include_directories(${id} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    gtest_discover_tests(${id})
    add_test(NAME ${id} COMMAND ${id})
endforeach()
