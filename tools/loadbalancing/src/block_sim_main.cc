#include "block_alloc_sim.h"

#include <climits>
#include <getopt.h>

amr::BlockSimulatorOpts options;

void PrintHelp(int argc, char* argv[]) {
  fprintf(stderr, "\n\tUsage: %s -p <profile_dir>\n", argv[0]);
  exit(-1);
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
  options.nranks = -1;
  options.nblocks = -1;

  while ((c = getopt(argc, argv, "b:c:e:hn:p:r:")) != -1) {
    switch (c) {
      case 'b':
        options.nblocks = atoi(optarg);
      case 'c':
        options.prof_time_combine_policy = optarg;
        break;
      case 'e':
        ParseCsvStr(optarg, options.events);
        break;
      case 'p':
        options.prof_dir = optarg;
        break;
      case 'r':
        options.nranks = atoi(optarg);
      case 'n':
        options.nts = atoi(optarg);
        break;
      case 'h':
        PrintHelp(argc, argv);
    }
  }

  pdlfs::Env* env = pdlfs::Env::Default();
  options.env = env;

  if (options.prof_dir.empty()) {
    logf(LOG_ERRO, "No profile_dir specified!");
    PrintHelp(argc, argv);
  }

  if (!options.env->FileExists(options.prof_dir.c_str())) {
    logf(LOG_ERRO, "Directory does not exist!!!");
    PrintHelp(argc, argv);
  }

  if (options.nblocks < 0) {
    logf(LOG_ERRO, "No nblocks specified!");
    PrintHelp(argc, argv);
  }

  if (options.nranks < 0) {
    logf(LOG_ERRO, "No nranks specified!");
    PrintHelp(argc, argv);
  }

  options.output_dir = options.prof_dir + "/block_sim";

  logf(LOG_INFO,
       "[Initial Parameters] nranks=%d, nblocks=%d, nts=%d\n"
       "output_dir=%s",
       options.nranks, options.nblocks, options.nts,
       options.output_dir.c_str());
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
