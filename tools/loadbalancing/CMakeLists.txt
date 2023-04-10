set(lb_srcs
  main.cc)

add_executable(lb-sim ${lb_srcs} ${common_srcs})
target_include_directories(lb-sim PRIVATE ${common_include})
