#include "block_alloc_sim.h"

#include <climits>
#include <getopt.h>

amr::BlockSimulatorOpts options;

void PrintHelp(int argc, char* argv[]) {
  fprintf(stderr, "\n\tUsage: %s -p <profile_dir>\n", argv[0]);
}

void ParseCsvStr(const char* str, std::vector<int>& vals) {
  vals.clear();
  int num, nb;

  while (sscanf(str, "%d%n", &num, &nb) >= 1) {
    vals.push_back(num);
    str += nb;
    if (str[0] != ',') break;
    str += 1;
  }

  logf(LOG_INFO, "BlockSim: parsed %zu events", vals.size());
}

void ParseOptions(int argc, char* argv[]) {
  extern char* optarg;
  extern int optind;
  int c;

  options.prof_dir = "";
  options.prof_time_combine_policy = "add";
  options.nts = INT_MAX;

  while ((c = getopt(argc, argv, "c:e:hn:p:")) != -1) {
    switch (c) {
      case 'c':
        options.prof_time_combine_policy = optarg;
        break;
      case 'e':
        ParseCsvStr(optarg, options.events);
        break;
      case 'p':
        options.prof_dir = optarg;
        break;
      case 'n':
        options.nts = atoi(optarg);
        break;
      case 'h':
        PrintHelp(argc, argv);
        exit(0);
    }
  }

  pdlfs::Env* env = pdlfs::Env::Default();
  options.env = env;
  options.nranks = 512;
  options.nblocks = 512;

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

  options.output_dir = options.prof_dir + "/block_sim";
}

void Run() {
  amr::BlockSimulator sim(options);
  sim.Run();
}

int main(int argc, char* argv[]) {
  ParseOptions(argc, argv);
  Run();
  return 0;
}
