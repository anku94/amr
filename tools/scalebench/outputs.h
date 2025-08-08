//
// Created by Ankush J on 7/18/23.
//

#pragma once

#include <pdlfs-common/env.h>

#include "policy.h"
#include "tabular_data.h"
#include "writable_file.h"

namespace amr {
class ScaleRow : public TableRow {
 private:
  std::vector<std::string> header = {"policy",    "nblocks", "nranks",
                                     "iter_time", "rt_avg",  "rt_max",
                                     "loc_cost"};
  std::vector<std::string> data;

 public:
  // clang-format off
  ScaleRow(const char* policy_name, int nblocks, int nranks, double iter_time,
           double rt_avg, double rt_max, double loc_cost)
      : data({
          policy_name,
          std::to_string(nblocks),
          std::to_string(nranks),
          FixedWidthString(iter_time, 2),
          FixedWidthString(rt_avg, 2),
          FixedWidthString(rt_max, 2),
          FixedWidthString(loc_cost, 2)
          }) {}
  // clang-format on

  std::vector<std::string> GetHeader() const override { return header; }

  std::vector<std::string> GetData() const override { return data; }

  static std::string FixedWidthString(double d, int n) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(n) << d;
    return ss.str();
  }
};

class ScaleExecLog : public WritableCSVFile {
 public:
  ScaleExecLog(pdlfs::Env* const env, const std::string& fpath)
      : WritableCSVFile(env, fpath) {}
  void WriteRow(const char* policy_name, int nblocks, int nranks,
                double iter_time, double rt_avg, double rt_max,
                double loc_cost) {
    std::string policy_name_cleaned =
        PolicyUtils::GetSafePolicyName(policy_name);

    char str[1024];
    int len = snprintf(str, 1024, "%s,%d,%d,%.2lf,%.2lf,%.2lf,%.2lf\n",
                       policy_name_cleaned.c_str(), nblocks, nranks, iter_time,
                       rt_avg, rt_max, loc_cost);
    assert(len < 1024);
    AppendRow(str, len);

    std::shared_ptr<TableRow> row = std::make_shared<ScaleRow>(
        policy_name, nblocks, nranks, iter_time, rt_avg, rt_max, loc_cost);
    table_.addRow(row);
  }

  std::string GetTabularStr() const {
    std::stringstream ss;
    table_.emitTable(ss);
    return ss.str();
  }

 private:
  void WriteHeader() override { return; }

 private:
  TabularData table_;
};
}  // namespace amr
