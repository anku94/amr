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
    if (str[nb] != ',') break;
  }

  logf(LOG_INFO, "Read %zu items", vals.size());
}

void ParseOptions(int argc, char* argv[]) {
  extern char* optarg;
  extern int optind;
  int c;

  options.prof_dir = "";
  options.nts = INT_MAX;

  while ((c = getopt(argc, argv, "e:hn:p:")) != -1) {
    switch (c) {
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
