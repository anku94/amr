find_package(pdlfs-common CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(lb_common_srcs
    alias_method.cc
    config_parser.cc
    globals.cc
)

add_library(lb_common STATIC ${lb_common_srcs})
target_include_directories(lb_common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(lb_common PUBLIC pdlfs-common glog::glog MPI::MPI_CXX)