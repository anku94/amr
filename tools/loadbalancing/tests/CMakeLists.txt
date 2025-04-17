find_package(GTest CONFIG)

if (NOT GTest_FOUND)
  return()
endif()

set(lb_tests 
  lb_tests.cc 
  lb_util_tests.cc
  lb_chunkwise_tests.cc 
  policy_tests.cc)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(all_include_dirs 
  ${src_dir} 
  ${src_dir}/blocksim 
  ${src_dir}/common)

set(misc_srcs
    ${src_dir}/blocksim/block_alloc_sim.cc
    ${src_dir}/blocksim/policy_exec_ctx.cc
    ${src_dir}/blocksim/policy_stats.cc
    )

foreach (lcv ${lb_tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} ${lcv})
    target_link_libraries(${id} GTest::gtest_main lb lb_common)
    target_include_directories(${id} PRIVATE ${all_include_dirs})
    gtest_discover_tests(${id})
    add_test(NAME ${id} COMMAND ${id})
endforeach()

add_executable (misc_tests misc_tests.cc ${misc_srcs})
target_link_libraries(misc_tests GTest::gtest_main lb lb_common)
target_include_directories(misc_tests PRIVATE ${all_include_dirs})
gtest_discover_tests(misc_tests)
add_test(NAME misc_tests COMMAND misc_tests)

add_executable (scale_tests scale_tests.cc)
target_link_libraries(scale_tests GTest::gtest_main lb lb_common)
target_include_directories(scale_tests PRIVATE ${all_include_dirs})
gtest_discover_tests(scale_tests)
add_test(NAME scale_tests COMMAND scale_tests)
