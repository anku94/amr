include(GoogleTest)

find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

find_package(glog REQUIRED)
# One target uses gflags - probably want to get rid of the dependency later
find_package(gflags)

add_executable(halo3d halo3d.cc)
add_executable(halo3d-26 halo3d-26.cc)
add_executable(sweep3d sweep3d.cc)
add_executable(incast incast.cc)

add_executable(block_tests block_tests.cc)
add_test(block_tests block_tests)
gtest_discover_tests(block_tests)

set(all_bin halo3d halo3d-26 sweep3d incast)

foreach (tgt ${all_bin})
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX glog::glog)
endforeach()

target_link_libraries(block_tests glog::glog GTest::gtest_main)

if (gflags_FOUND)
  message(STATUS "gflags found. Building halo3d_v2.")
  add_executable(halo3d_v2 halo3d_v2.cc)
  target_link_libraries(halo3d_v2 PRIVATE MPI::MPI_CXX glog::glog gflags)
else()
  message(STATUS "gflags not found. Not building halo3d_v2.")
endif()

