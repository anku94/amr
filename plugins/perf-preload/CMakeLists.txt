find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(dl REQUIRED)

add_library(prof_preload SHARED example.cc)
target_link_libraries(prof_preload PRIVATE MPI::MPI_CXX Kokkos::kokkos dl)

set_target_properties(prof_preload PROPERTIES 
    LINK_FLAGS "-Wl,--no-as-needed")
