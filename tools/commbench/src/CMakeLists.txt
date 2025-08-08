find_package(pdlfs-common CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(MPI MODULE REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(topo_srcs
    bench/driver.cc
    bench/graph.cc
    bench/logger.cc
    bench/mesh_gen.cc
    bench/trace_reader.cc
    bench/single_ts_trace_reader.cc
    bench/mesh_driver.cc
    bench/newlogger.cc)

set(amr_srcs amr/bvar.cc amr/mesh.cc)

add_library(topo ${common_srcs} ${topo_srcs} ${amr_srcs})
target_include_directories(topo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                       "${common_include}")
target_link_libraries(topo PUBLIC lb tools-common)
target_link_libraries(topo PUBLIC pdlfs-common glog::glog MPI::MPI_CXX)

add_executable(meshdriver_main meshdriver_main.cc)
target_link_libraries(meshdriver_main topo)

# Mac doesn't set @rpath without this
set_target_properties(meshdriver_main PROPERTIES INSTALL_RPATH
                                                 "@executable_path/../lib")

install(TARGETS meshdriver_main RUNTIME DESTINATION bin)
