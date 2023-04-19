#include "policy_sim.h"

#include <getopt.h>

amr::PolicySimOptions options;

void PrintHelp(int argc, char* argv[]) {
  fprintf(stderr, "\n\tUsage: %s -i <profile_dir>\n", argv[0]);
}

void ParseOptions(int argc, char* argv[]) {
  extern char* optarg;
  extern int optind;
  int c;

  options.prof_dir = "";
while ((c = getopt(argc, argv, "hp:i:n:t:")) != -1) {
    switch (c) {
      case 'p':
        options.prof_dir = optarg;
        break;
      case 'i':
        options.ilp_shard_idx = strtol(optarg, nullptr, 10);
        options.sim_ilp = true;
        break;
      case 'n':
        options.ilp_num_shards = strtol(optarg, nullptr, 10);
        options.sim_ilp = true;
        break;
      case 't':
        options.num_ts = strtol(optarg, nullptr, 10);
        options.sim_ilp = true;
        break;
      case 'h':
        PrintHelp(argc, argv);
        exit(0);
    }
  }

  pdlfs::Env* env = pdlfs::Env::Default();
  options.env = env;
  options.nranks = 512;

  if (options.prof_dir.empty()) {
    logf(LOG_ERRO, "No profile_dir specified!");
    PrintHelp(argc, argv);
    exit(-1);
  }

  if (!options.env->FileExists(options.prof_dir.c_str())) {
    logf(LOG_ERRO, "Directory does not exist!!!");
    PrintHelp(argc, argv);
    exit(-1);
  }

  if (options.sim_ilp) {
    if (options.ilp_shard_idx < 0 || options.ilp_num_shards < 0 ||
        options.num_ts < 0) {
      logf(LOG_ERRO, "Invalid ILP options!");
      PrintHelp(argc, argv);
      exit(-1);
    } else {
      logf(LOG_INFO, "[PolicySim] Simulating ILP only: %d/%d",
           options.ilp_shard_idx, options.ilp_num_shards);
    }
  } else {
    logf(LOG_INFO, "[PolicySim] Simulating all timesteps");
  }

  options.output_dir = options.prof_dir + "/lb_sim";
}

void Run() {
  amr::PolicySim sim(options);
  sim.Run();
}

int main(int argc, char* argv[]) {
  ParseOptions(argc, argv);
  Run();
  return 0;
}
