find_package(pdlfs-common REQUIRED)
# find_package(lb REQUIRED)

set(blocksim_srcs
    block_alloc_sim.cc
    policy_exec_ctx.cc
    policy_stats.cc)

message(STATUS "Blocksim should include: ${CMAKE_CURRENT_SOURCE_DIR}/../common")

add_executable(block-sim block_sim_main.cc ${blocksim_srcs})
target_link_libraries(block-sim pdlfs-common lb)
target_include_directories(block-sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(simple-sim simple_sim.cc)
target_link_libraries(simple-sim pdlfs-common lb)
target_include_directories(simple-sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)

install(TARGETS block-sim DESTINATION bin)
install(TARGETS simple-sim DESTINATION bin)
