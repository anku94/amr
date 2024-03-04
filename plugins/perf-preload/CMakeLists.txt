find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

add_library(prof_preload SHARED example.cc)
target_link_libraries(prof_preload PRIVATE MPI::MPI_CXX Kokkos::kokkos ${CMAKE_DL_LIBS})

set_target_properties(prof_preload PROPERTIES 
    LINK_FLAGS "-Wl,--no-as-needed")

add_executable(example_prog example_prog.cc)
target_link_libraries(example_prog PRIVATE MPI::MPI_CXX Kokkos::kokkos)
