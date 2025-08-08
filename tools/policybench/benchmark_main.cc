#include "benchmark.h"

#include <unistd.h>

/* Process:
 * 1. Run() in benchmark.h calls RunCppIterSuite()
 * 2. A vector<RunType> is generated, and initialized with all policies.
 * 3. DoRuns() is called.
 */

std::string config_file;
std::string output_dir;

// Expect -c <config_file> -o <output_dir>, error otherwise
void GetConfig(int argc, char* argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s -c <config_file> -o <output_dir>\n", argv[0]);
    exit(1);
  }

  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-c") == 0) {
      config_file = argv[i + 1];
    } else if (strcmp(argv[i], "-o") == 0) {
      output_dir = argv[i + 1];
    } else {
      fprintf(stderr, "Usage: %s -c <config_file> -o <output_dir>\n", argv[0]);
      exit(1);
    }
  }

  // assert config_file exists and is a file
  // if (access(config_file.c_str(), F_OK) == -1) {
  //   fprintf(stderr, "Config file %s does not exist\n", config_file.c_str());
  //   exit(1);
  // }
}

void Run() {
  amr::BenchmarkOpts opts{pdlfs::Env::Default(), output_dir};
  amr::Benchmark benchmark(opts);
  // benchmark.Run();
  benchmark.RunSuiteMini();
}

int main(int argc, char* argv[]) {
  GetConfig(argc, argv);
  Run();
  return 0;
}
