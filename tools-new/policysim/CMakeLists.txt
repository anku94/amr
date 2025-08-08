find_package(pdlfs-common REQUIRED)

set(blocksim_srcs block_alloc_sim.cc policy_exec_ctx.cc policy_stats.cc)

add_executable(policysim block_sim_main.cc ${blocksim_srcs})
target_link_libraries(policysim pdlfs-common lb tools-common)

add_executable(simplesim simple_sim.cc)
target_link_libraries(simplesim pdlfs-common lb tools-common)

add_executable(policysim-test policysim_tests.cc ${blocksim_srcs})
target_link_libraries(policysim-test pdlfs-common lb tools-common
                      GTest::gtest_main)

install(TARGETS policysim DESTINATION bin)
install(TARGETS simplesim DESTINATION bin)
install(TARGETS policysim-test DESTINATION bin)