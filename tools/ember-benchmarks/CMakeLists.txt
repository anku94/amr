include(GoogleTest)

add_executable(halo3d halo3d.cc)
add_executable(halo3d-26 halo3d-26.cc)
add_executable(sweep3d sweep3d.cc)
add_executable(incast incast.cc)
add_executable(halo3d_v2 halo3d_v2.cc)

add_executable(block_tests block_tests.cc)
add_test(block_tests block_tests)
gtest_discover_tests(block_tests)

find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

find_package(glog REQUIRED)

set(all_bin halo3d halo3d-26 sweep3d incast halo3d_v2)

foreach (tgt ${all_bin})
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX glog::glog)
endforeach()

target_link_libraries(block_tests glog::glog GTest::gtest_main)
