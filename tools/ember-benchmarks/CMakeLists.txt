add_executable(halo3d halo3d.cc)
add_executable(halo3d-26 halo3d-26.cc)
add_executable(sweep3d sweep3d.cc)
add_executable(incast incast.cc)
add_executable(halo3d_v2 halo3d_v2.cc)
add_executable(test test.cc)

find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

#
# make sure we link with MPI.  use "MPI_CXX_COMPILE_FLAGS_LIST"
# prepared by the calling module.
#
foreach (tgt halo3d halo3d-26 sweep3d incast halo3d_v2 test)

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
