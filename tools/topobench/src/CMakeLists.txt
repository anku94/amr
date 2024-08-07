find_package(pdlfs-common CONFIG REQUIRED)

#add_executable(topobench topobench.cc block.cc driver.cc logger.cc topology.cc common.cc logger.h logger.cc)
set(topo_srcs
        bvar.cc
        driver.cc
        globals.cc
        graph.cc
        logger.cc
        mesh_gen.cc
        trace_reader.cc
        single_ts_trace_reader.cc
      )

add_library(topo ${topo_srcs} ${common_srcs})
target_include_directories(topo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${common_include}")
target_link_libraries(topo pdlfs-common)

add_executable(topobench topobench.cc)
target_link_libraries(topobench topo)

# Find required packages
find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

#
# make sure we link with MPI.  use "MPI_CXX_COMPILE_FLAGS_LIST"
# prepared by the calling module.
#
foreach (tgt topo topobench)

    # mpich on ub14 gives a leading space that we need to trim off
    foreach (lcv ${MPI_CXX_COMPILE_FLAGS_LIST})
        if (NOT ${lcv} STREQUAL "")
            target_compile_options (${tgt}
                    PUBLIC $<BUILD_INTERFACE:${lcv}>)
        endif ()
    endforeach ()

    # XXX: have to do this one dir at a time
    foreach (lcv ${MPI_CXX_INCLUDE_PATH})
        target_include_directories (${tgt}
                PUBLIC $<BUILD_INTERFACE:${lcv}>)
    endforeach ()

    foreach (lcv ${MPI_CXX_LIBRARIES})
        target_link_libraries(${tgt} $<BUILD_INTERFACE:${lcv}>)
    endforeach ()

    set_property (TARGET ${tgt} APPEND PROPERTY LINK_FLAGS
            ${MPI_CXX_LINK_FLAGS})

endforeach ()

install(TARGETS topobench RUNTIME DESTINATION bin)
