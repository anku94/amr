#include "driver.h"

#include <getopt.h>
#include <glog/logging.h>

void PrintHelp(const char *exec) {
  printf("Usage: %s\n"
         "\t-j <job_dir for output> \n"
         "\t-n <num_ts>\n"
         "\t-b <blocks_per_rank>\n"
         "\t-r <num_rounds> \n"
         "\t-s <msg_sz>\n"
         "\t-t <topology:12345>\n"
         "\t-p <trace_path if 4|5>\n",
         exec);

  printf("\n\tTopology:\n"
         "\t1: Ring\n"
         "\t2: AllToAll\n"
         "\t3: Dynamic\n"
         "\t4: FromSingleTSTrace\n"
         "\t5: FromMultiTSTrace\n");

  printf("\nIf using -t 4, -n num_ts is optional, defaults to 1\n"
         "If using -t 5, -n num_ts is optional, defaults to 2\n"
         "For both -t 4 and -t 5, -b and -s are ignored\n");
}

MeshGenMethod parse_method(int topo_id) {
  switch (topo_id) {
  case 1:
    return MeshGenMethod::Ring;
  case 2:
    return MeshGenMethod::AllToAll;
  case 3:
    return MeshGenMethod::Dynamic;
  case 4:
    return MeshGenMethod::FromSingleTSTrace;
  case 5:
    return MeshGenMethod::FromMultiTSTrace;
  }

  return MeshGenMethod::Ring;
}

void parse_opts(int argc, char *argv[], DriverOpts &opts) {
  int c;
  extern char *optarg;
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
      opts.meshgen_method = parse_method(std::stoi(optarg));
      break;
    default:
      PrintHelp(argv[0]);
      break;
    }
  }

  opts.bench_log = "topobench.csv";

  if (!opts.IsValid()) {
    PrintHelp(argv[0]);
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  DriverOpts opts;
  parse_opts(argc, argv, opts);
  Driver driver(opts);
  driver.Run(argc, argv);
  return 0;
}
