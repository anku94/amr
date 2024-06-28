#include "driver.h"
#include "file_utils.h"

#include <getopt.h>
#include <glog/logging.h>

void PrintHelp() {
  printf(
      "\n\t./prog -j <job_dir> \n"
      " -b <blocks_per_rank> -r <num_rounds> \n"
      " -s <msg_sz> -t <topology:1234> -p <prof_dir>\n");
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
      return NeighborTopology::FromSingleTSTrace;
    case 5:
      return NeighborTopology::FromMultiTSTrace;
  }

  return NeighborTopology::Ring;
}

void parse_opts(int argc, char* argv[], DriverOpts& opts) {
  int c;
  extern char* optarg;
  extern int optind;

  while ((c = getopt(argc, argv, "b:j:n:p:r:s:t:")) != -1) {
    switch (c) {
      case 'b':
        opts.blocks_per_rank = std::stoi(optarg);
        break;
      case 'j':
        opts.job_dir = optarg;
        break;
      case 'n':
        opts.comm_nts = std::stoi(optarg);
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

  opts.bench_log = "topobench.csv";

  if (!opts.IsValid()) {
    PrintHelp();
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  DriverOpts opts;
  parse_opts(argc, argv, opts);
  Driver driver(opts);
  driver.Run(argc, argv);
  return 0;
}
