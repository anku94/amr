#include <iostream>

#include "driver.h"

int main(int argc, char *argv[]) {
  DriverOpts opts;
  opts.blocks_per_rank = 4;
  opts.msgs_per_block = 4;
  opts.size_per_msg = 4096;
  Driver driver(opts);
  driver.Run(argc, argv);
  return 0;
}
