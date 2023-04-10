find_package(pdlfs-common CONFIG REQUIRED)

set(lb_srcs
        main.cc
        policy_sim.cc
        lb_policies.cc)

add_executable(lb-sim ${lb_srcs} ${common_srcs})
target_include_directories(lb-sim PRIVATE ${common_include})
target_link_libraries(lb-sim pdlfs-common)