find_package(pdlfs-common REQUIRED)
find_package(lb REQUIRED)

set(blocksim_srcs
    block_alloc_sim.cc
    policy_exec_ctx.cc
    policy_stats.cc)

add_executable(block-sim block_sim_main.cc ${blocksim_srcs})
target_link_libraries(block-sim pdlfs-common lb fort)

install(TARGETS block-sim DESTINATION bin)
