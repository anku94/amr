#include "benchmark.h"
#include "tabular_data.h"

void Run() {
  const char* dir_out = "/Users/schwifty/Repos/amr-data/20231114-policy-bench";
  amr::BenchmarkOpts opts{pdlfs::Env::Default(), dir_out};
  amr::Benchmark benchmark(opts);
  benchmark.Run();
}

int main() {
  //  TestTabularData();
  Run();
  return 0;
}