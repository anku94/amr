#include "driver.h"

#include <iostream>

int main(int argc, char* argv[]) {
  DriverOpts opts;
  opts.topology = NeighborTopology::Ring;
  opts.blocks_per_rank = 1;
  opts.msgs_per_block = 1;
  opts.size_per_msg = 4096;
  Driver driver(opts);
  driver.Run(argc, argv);
  return 0;
}
