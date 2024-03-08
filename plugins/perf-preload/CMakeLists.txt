find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(pdlfs-common REQUIRED)

set(PRELOAD_SRCS
    amr_monitor.cc
    common.cc
    kokkos_hooks.cc
    logging.cc
    mpi_hooks.cc
    p2p.cc)

add_library(prof_preload SHARED ${PRELOAD_SRCS})
target_link_libraries(prof_preload PRIVATE MPI::MPI_CXX Kokkos::kokkos ${CMAKE_DL_LIBS} pdlfs-common)

set_target_properties(prof_preload PROPERTIES 
    LINK_FLAGS "-Wl,--no-as-needed")

add_executable(example_prog example_prog.cc)
target_link_libraries(example_prog PRIVATE MPI::MPI_CXX Kokkos::kokkos)
