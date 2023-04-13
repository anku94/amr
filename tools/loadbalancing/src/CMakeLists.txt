find_package(pdlfs-common CONFIG REQUIRED)
set(lb_srcs
        lb_lspt.cc
        policy_sim.cc
        lb_policies.cc)

add_library(lb ${lb_srcs} ${common_srcs})
# target_include_directories(lb PUBLIC ${common_include})
target_include_directories(lb PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${common_include}>)
target_link_libraries(lb pdlfs-common)

add_executable(lb-sim main.cc)
target_link_libraries(lb-sim lb)
