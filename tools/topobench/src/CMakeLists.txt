find_package(pdlfs-common CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package (MPI REQUIRED)

set(topo_srcs
        bench/bvar.cc
        bench/driver.cc
        bench/globals.cc
        bench/graph.cc
        bench/logger.cc
        bench/mesh_gen.cc
        bench/trace_reader.cc
        bench/single_ts_trace_reader.cc
      )

add_library(topo ${topo_srcs} ${common_srcs})
target_include_directories(topo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${common_include}")
target_link_libraries(topo pdlfs-common glog::glog gflags)

add_executable(topobench bench/topobench.cc)
target_link_libraries(topobench topo MPI::MPI_CXX)

# Find required packages
# string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")
# target_link_libraries(topobench PUBLIC MPI::MPI_CXX)

install(TARGETS topobench RUNTIME DESTINATION bin)
