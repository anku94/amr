find_package(pdlfs-common CONFIG REQUIRED)

include(GoogleTest)

set(lb_srcs
        lb_lspt.cc
        policy_sim.cc
        lb_policies.cc)

add_library(lb ${lb_srcs} ${common_srcs})
target_include_directories(lb PUBLIC ${common_include})
target_link_libraries(lb pdlfs-common)

add_executable(lb-sim main.cc)
target_link_libraries(lb-sim lb)

set(lb_tests lb_tests.cc misc_tests.cc)
foreach (lcv ${lb_tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} EXCLUDE_FROM_ALL ${lcv})
    target_link_libraries(${id} GTest::gtest_main lb)
    gtest_discover_tests(${id})
endforeach()
