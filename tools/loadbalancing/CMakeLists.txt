set(lb-srcs
  main.cc
../common/common.cc)

add_executable(lb-sim ${lb-srcs})
target_include_directories(lb-sim PRIVATE ../common)
