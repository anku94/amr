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

  options.prof_dir = nullptr;

  while ((c = getopt(argc, argv, "i:h")) != -1) {
    switch (c) {
      case 'i':
        options.prof_dir = optarg;
        break;
      case 'h':
        PrintHelp(argc, argv);
        exit(0);
    }
  }

  pdlfs::Env* env = pdlfs::Env::Default();
  options.env = env;

  if (!options.prof_dir) {
    logf(LOG_ERRO, "No profile_dir specified!");
    PrintHelp(argc, argv);
    exit(-1);
  }

  if ((!options.prof_dir) || (!options.env->FileExists(options.prof_dir))) {
    logf(LOG_ERRO, "Directory does not exist!!!");
    PrintHelp(argc, argv);
    exit(-1);
  }
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
