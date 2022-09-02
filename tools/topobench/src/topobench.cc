#include "driver.h"

#include <getopt.h>
#include <iostream>

void PrintHelp() {
  printf(
      "\n\t./prog -b <blocks_per_rank> -r <num_rounds>"
      " -s <msg_sz> -t <topology:1234> -p <trace_root>\n");
}

NeighborTopology parse_topology(int topo_id) {
  switch (topo_id) {
    case 1:
      return NeighborTopology::Ring;
    case 2:
      return NeighborTopology::AllToAll;
    case 3:
      return NeighborTopology::Dynamic;
    case 4:
      return NeighborTopology::FromTrace;
  }

  return NeighborTopology::Ring;
}

void parse_opts(int argc, char* argv[], DriverOpts& opts) {
  int c;
  extern char* optarg;
  extern int optind;

  while ((c = getopt(argc, argv, "b:p:r:s:t:")) != -1) {
    switch (c) {
      case 'b':
        opts.blocks_per_rank = std::stoi(optarg);
        break;
      case 'p':
        opts.trace_root = optarg;
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

  if (!opts.IsValid()) {
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
