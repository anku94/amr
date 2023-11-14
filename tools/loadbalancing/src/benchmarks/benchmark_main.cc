#include "benchmark.h"
#include "tabular_data.h"

void TestTabularData() {
  amr::TabularData table;

  // Initialize CustomRows with an initializer list syntax
  std::shared_ptr<amr::Row> row1 = std::make_shared<amr::CustomRow>(1, "abcd", 2.0f);
  table.addRow(row1);

  // ...add more rows as needed

  // Print table to stdout
  table.emitTable(std::cout);

  // Get CSV string
  std::string csvData = table.toCSV();
  std::cout << csvData;
}

void Run() {
  amr::BenchmarkOpts opts;
  opts.nranks = 512;
  opts.nblocks = 2000;
  amr::Benchmark benchmark(opts);
  benchmark.Run();
//  benchmark.Exp();
}

int main() {
//  TestTabularData();
  Run();
  return 0;
}