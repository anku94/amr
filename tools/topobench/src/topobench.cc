#include "driver.h"

#include <getopt.h>
#include <iostream>

void PrintHelp() {
  printf(
      "\n\t./prog -b <blocks_per_rank> -r <num_rounds>"
      " -s <msg_sz> -t <topology>\n");
}

NeighborTopology parse_topology(int topo_id) {
  switch (topo_id) {
    case 1:
      return NeighborTopology::Ring;
    case 2:
      return NeighborTopology::AllToAll;
  }

  return NeighborTopology::Ring;
}

void parse_opts(int argc, char* argv[], DriverOpts& opts) {
  int c;
  extern char* optarg;
  extern int optind;

  opts.blocks_per_rank = SIZE_MAX;
  opts.size_per_msg = SIZE_MAX;
  opts.comm_rounds = SIZE_MAX;

  while ((c = getopt(argc, argv, "b:r:s:t:")) != -1) {
    switch (c) {
      case 'b':
        opts.blocks_per_rank = std::stoi(optarg);
        break;
      case 'r':
        opts.comm_rounds = std::stoi(optarg);
        break;
      case 's':
        opts.size_per_msg = std::stoi(optarg);
        break;
      case 't':
        opts.topology = parse_topology(std::stoi(optarg));
        break;
      default:
        PrintHelp();
        break;
    }
  }

  if ((opts.blocks_per_rank == SIZE_MAX) || (opts.size_per_msg == SIZE_MAX) ||
      (opts.comm_rounds == SIZE_MAX)) {
    PrintHelp();
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  DriverOpts opts;
  parse_opts(argc, argv, opts);
  Driver driver(opts);
  driver.Run(argc, argv);
  return 0;
}
