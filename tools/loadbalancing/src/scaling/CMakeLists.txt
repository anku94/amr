find_package(pdlfs-common REQUIRED)

set(scalesim_srcs
  scale_sim.cc
  scale_sim_main.cc
  ../globals.cc
  ../config_parser.cc
  ../distrib/alias_method.cc
)

add_executable(scale-sim ${scalesim_srcs})
target_link_libraries(scale-sim pdlfs-common lb)

install(TARGETS scale-sim DESTINATION bin)
