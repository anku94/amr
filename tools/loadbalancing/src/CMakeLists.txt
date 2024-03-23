find_package(pdlfs-common CONFIG REQUIRED)
find_package(GUROBI)

# From Gurobi's example CMakeLists... not sure if thing thing works with 14
# We're essentially overriding the top-level directive... do something else?
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(benchmarks)

set(lb_srcs
        policies/lb_contig_improv.cc
        policies/lb_ilp.cc
        policies/lb_lspt.cc
        policies/lb_cpp_iter.cc
        block_alloc_sim.cc
        policy.cc
        policy_exec_ctx.cc
        lb_policies.cc
        scaling/scale_sim.cc)

add_library(lb ${lb_srcs} ${common_srcs})

# if GUROBI is found, set macro GUROBI_ENABLED
if (GUROBI_FOUND)
    add_definitions(-DGUROBI_ENABLED)
    target_compile_definitions(lb PRIVATE GUROBI_ENABLED)
endif()

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

if (GUROBI_FOUND)
    target_compile_definitions(lb PRIVATE GUROBI_ENABLED)
    target_link_libraries(lb PUBLIC ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    # FindGUROBI does not unify both these into a single target :(
    target_include_directories(lb PRIVATE ${GUROBI_INCLUDE_DIRS})
endif()

add_executable(block-sim block_sim_main.cc)
target_link_libraries(block-sim PRIVATE lb pdlfs-common fort)

add_executable(scale-sim scaling/scale_sim_main.cc)
target_link_libraries(scale-sim PRIVATE lb pdlfs-common fort)

add_executable(cluster-sim cluster_main.cc)
target_link_libraries(cluster-sim PRIVATE lb pdlfs-common fort)


#
# "make install" rules
#
install(TARGETS lb EXPORT lb
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(TARGETS block-sim scale-sim cluster-sim
        RUNTIME DESTINATION bin)
install(EXPORT lb
        FILE lb-config.cmake
        DESTINATION lib/cmake/lb)
install(DIRECTORY ../include
        DESTINATION .
        FILES_MATCHING PATTERN "*.h")
