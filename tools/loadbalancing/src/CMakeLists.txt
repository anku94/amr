find_package(pdlfs-common CONFIG REQUIRED)
find_package(GUROBI REQUIRED)

# From Gurobi's example CMakeLists... not sure if thing thing works with 14
# We're essentially overriding the top-level directive... do something else?
set(CMAKE_CXX_STANDARD 11)

set(lb_srcs
        block_alloc_sim.cc
        lb_ilp.cc
        lb_contig_improv.cc
        lb_lspt.cc
        policy.cc
        policy_exec_ctx.cc
        policy_sim.cc
        lb_policies.cc)

add_library(lb ${lb_srcs} ${common_srcs})

# GUROBI requires C++11; *sigh*
target_compile_features(lb PRIVATE cxx_std_11)

target_include_directories(lb PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${common_include}>
        $<INSTALL_INTERFACE:include>
        )

#target_link_libraries(lb PRIVATE ${GUROBI_CXX_LIBRARY})
target_link_libraries(lb PRIVATE pdlfs-common fort)
target_link_libraries(lb PUBLIC ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})

# FindGUROBI does not unify both these into a single target :(
target_include_directories(lb PRIVATE ${GUROBI_INCLUDE_DIRS})

add_executable(lb-sim policy_main.cc)
#target_link_libraries(lb-sim lb ${GUROBI_LIBRARY} ${GUROBI_CXX_LIBRARY})
target_link_libraries(lb-sim PRIVATE lb pdlfs-common fort)

add_executable(block-sim block_sim_main.cc)
#target_link_libraries(lb-sim lb ${GUROBI_LIBRARY} ${GUROBI_CXX_LIBRARY})
target_link_libraries(block-sim PRIVATE lb pdlfs-common fort)

add_executable(cluster-sim cluster_main.cc)
target_link_libraries(cluster-sim PRIVATE lb pdlfs-common fort)

#
# "make install" rules
#
install(TARGETS lb EXPORT lb
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(EXPORT lb
        FILE lb-config.cmake
        DESTINATION lib/cmake/lb)
install(DIRECTORY ../include
        DESTINATION .
        FILES_MATCHING PATTERN "*.h")
