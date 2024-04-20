//
// Created by Ankush J on 4/17/23.
//

#include "cluster_sim.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <getopt.h>

amr::ClusterSimOptions options;

void PrintHelp(int argc, char* argv[]) {
  fprintf(stderr, "\n\tUsage: %s -i <profile_dir>\n", argv[0]);
}

void ParseOptions(int argc, char* argv[]) {
  extern char* optarg;
  extern int optind;
  int c;

  while ((c = getopt(argc, argv, "hp:")) != -1) {
    switch (c) {
      case 'p':
        options.prof_dir = optarg;
        break;
      case 'h':
        PrintHelp(argc, argv);
        exit(0);
    }
  }

  pdlfs::Env* env = pdlfs::Env::Default();
  options.env = env;

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

  options.output_dir = options.prof_dir + "/cluster_sim";
}

void Run() {
  amr::ClusterSim sim(options);
  sim.Run();
}

int main(int argc, char* argv[]) {
  ParseOptions(argc, argv);
  Run();
  return 0;
}