#include "benchmark.h"

/* Process:
 * 1. Run() in benchmark.h calls RunCppIterSuite()
 * 2. A vector<RunType> is generated, and initialized with all policies.
 * 3. DoRuns() is called.
 */

void Run() {
  const char* dir_out = "/l0/amr-bench-out";
  amr::BenchmarkOpts opts{pdlfs::Env::Default(), dir_out};
  amr::Benchmark benchmark(opts);
  benchmark.Run();
}

int main() {
  Run();
  return 0;
}
