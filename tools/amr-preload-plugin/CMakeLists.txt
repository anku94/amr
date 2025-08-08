find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(pdlfs-common REQUIRED)

set(PRELOAD_SRCS
    amr_monitor.cc
    common.cc
    detailed_logger.cc
    kokkos_hooks.cc
    mpi_hooks.cc
    p2p.cc)

add_library(amr_preload SHARED ${PRELOAD_SRCS})
target_link_libraries(amr_preload PUBLIC MPI::MPI_CXX Kokkos::kokkos
                                         ${CMAKE_DL_LIBS} pdlfs-common tools-common)

#set_target_properties(amr_preload PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
# Tell linker to not remove unused symbols
if(APPLE)
    set_target_properties(amr_preload PROPERTIES 
        LINK_FLAGS "-Wl,-all_load")
else()
    set_target_properties(amr_preload PROPERTIES 
        LINK_FLAGS "-Wl,--whole-archive")
endif()

add_executable(amr_preload_example example_prog.cc)

target_include_directories(
  amr_preload_example
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tools/loadbalancing/include>)
target_link_libraries(amr_preload_example PRIVATE amr_preload)
# target_link_libraries(amr_preload_example PRIVATE MPI::MPI_CXX Kokkos::kokkos)

install(TARGETS amr_preload DESTINATION lib)
install(TARGETS amr_preload_example DESTINATION bin)
