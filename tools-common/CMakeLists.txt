find_package(pdlfs-common CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(tools-common-srcs
    alias_method.cc
    config_parser.cc
    globals.cc
)

add_library(tools-common STATIC ${tools-common-srcs})
target_include_directories(tools-common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(tools-common PUBLIC pdlfs-common glog::glog MPI::MPI_CXX)