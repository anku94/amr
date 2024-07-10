//
// Created by Ankush J on 7/13/23.
//

#include <getopt.h>

#include "globals.h"
#include "scale_sim.h"

amr::ScaleSimOpts options;

void PrintHelp(int argc, char* argv[]) {
  fprintf(stderr, "Usage: %s [options] \n", argv[0]);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -o <output_dir>   Output directory\n");
  fprintf(stderr, "  -r <nranks>       Number of ranks (opt)\n");
  fprintf(stderr, "  -s <block_beg>    Block size begin\n");
  fprintf(stderr, "  -e <block_end>    Block size end\n");
  fprintf(stderr, "  -h                Print this help message\n");
  exit(1);
}

void ParseOptions(int argc, char* argv[]) {
  extern char* optarg;
  extern int optind;
  int c;
  options.nblocks_beg = -1;
  options.nblocks_end = -1;
  options.nranks = -1;
  options.output_dir = "";

  while ((c = getopt(argc, argv, "e:ho:r:s:")) != -1) {
    switch (c) {
      case 'e':
        options.nblocks_end = atoi(optarg);
        break;
      case 'h':
        PrintHelp(argc, argv);
        break;
      case 'o':
        options.output_dir = optarg;
        break;
      case 'r':
        options.nranks = atoi(optarg);
        break;
      case 's':
        options.nblocks_beg = atoi(optarg);
        break;
    }
  }

  options.env = pdlfs::Env::Default();

  if (options.output_dir.empty()) {
    logv(__LOG_ARGS__, LOG_ERRO, "Output directory not specified\n");
    PrintHelp(argc, argv);
  }

  if (!options.env->FileExists(options.output_dir.c_str())) {
    logv(__LOG_ARGS__, LOG_ERRO, "Output directory does not exist\n");
    PrintHelp(argc, argv);
  }

  if (options.nblocks_beg < 0 || options.nblocks_end < 0) {
    logv(__LOG_ARGS__, LOG_ERRO, "Block size not specified\n");
    PrintHelp(argc, argv);
  }
}

int main(int argc, char* argv[]) {
  ParseOptions(argc, argv);
  amr::Globals.config = std::make_unique<amr::ConfigParser>();
  amr::ScaleSim sim(options);

  MPI_Init(&argc, &argv);

  int mympirank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mympirank);

  int nmpiranks = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &nmpiranks);

  if (nmpiranks == 1) {
    logv(__LOG_ARGS__, LOG_INFO, "Single rank, running Serial");
    sim.Run();
  } else {
    if (mympirank == 0) {
      logv(__LOG_ARGS__, LOG_INFO, "Running Parallel with %d ranks", nmpiranks);
      logv(__LOG_ARGS__, LOG_INFO,
           "Warning: AssignBlocks can not tell the difference between policy "
           "nranks and computation nranks, so make sure you have enough.");
    }

    sim.RunParallel(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
