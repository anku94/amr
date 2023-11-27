//
// Created by Ankush J on 11/16/23.
//

#include "common.h"
#include "tabular_data.h"

#include <gtest/gtest.h>

namespace amr {
class BenchmarkTest : public ::testing::Test {
 protected:
};

class CustomRow : public TableRow {
 private:
  int a;
  std::string b;
  float c;
  std::vector<std::string> header{"A", "B", "C"};

 public:
  CustomRow(int a, std::string b, float c) : a(a), b(std::move(b)), c(c) {}

  std::vector<std::string> GetHeader() const override { return header; }

  std::vector<std::string> GetData() const override {
    return {std::to_string(a), b, std::to_string(c)};
  }
};

TEST_F(BenchmarkTest, BasicTest) { logf(LOG_INFO, "HelloWorld!\n"); }

TEST_F(BenchmarkTest, TabularTest) {
  TabularData table;

  std::shared_ptr<TableRow> row1 =
      std::make_shared<CustomRow>(1, "abcd", 2.0f);
  table.addRow(row1);

  table.emitTable(std::cout);
  std::string csvData = table.toCSV();
  ASSERT_STRCASEEQ(csvData.c_str(), "A,B,C\n1,abcd,2.000000\n");
}
}  // namespace amr
