find_package(MPI REQUIRED)
find_package(pdlfs-common REQUIRED)

set(scalesim_srcs
  scale_sim.cc
  scale_sim_main.cc
)

add_executable(scale-sim ${scalesim_srcs})
target_link_libraries(scale-sim lb tools-common MPI::MPI_CXX)
# target_include_directories(scale-sim PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/../common)

install(TARGETS scale-sim DESTINATION bin)
