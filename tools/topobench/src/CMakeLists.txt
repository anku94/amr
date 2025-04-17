find_package(pdlfs-common CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package (MPI MODULE REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(topo_srcs
        bench/driver.cc
        bench/graph.cc
        bench/logger.cc
        bench/mesh_gen.cc
        bench/trace_reader.cc
        bench/single_ts_trace_reader.cc
        bench/mesh_driver.cc
      )

set (amr_srcs 
    amr/bvar.cc
    amr/globals.cc
    amr/mesh.cc)


add_library(topo ${topo_srcs} ${common_srcs} ${amr_srcs})
target_include_directories(topo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${common_include}")
target_link_libraries(topo PUBLIC pdlfs-common glog::glog MPI::MPI_CXX)

# add_executable(topobench bench/topobench.cc)
# target_link_libraries(topobench topo)

add_executable(meshdriver_main meshdriver_main.cc ${amr_srcs})
target_link_libraries(meshdriver_main topo)

# install(TARGETS topobench RUNTIME DESTINATION bin)
