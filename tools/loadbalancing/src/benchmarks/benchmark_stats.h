//
// Created by Ankush J on 11/9/23.
//

#pragma once

#include "tabular_data.h"

#include <utility>

namespace amr {
class BenchmarkRow : public TableRow {
 private:
  int nranks_;
  int nblocks_;
  std::string policy_name_;
  std::string distrib_name_;
  double time_avg_;
  double time_max_;

  std::vector<std::string> header = {"nranks", "nblocks",  "distrib",
                                     "policy", "time_avg", "time_max"};

 public:
  BenchmarkRow(int nranks, int nblocks, std::string distrib_name,
               std::string policy_name, double time_avg, double time_max)
      : nranks_(nranks),
        nblocks_(nblocks),
        distrib_name_(std::move(distrib_name)),
        policy_name_(std::move(policy_name)),
        time_avg_(time_avg),
        time_max_(time_max) {}

  std::vector<std::string> GetHeader() const override { return header; }

  std::vector<std::string> GetData() const override {
    return {std::to_string(nranks_),
            std::to_string(nblocks_),
            distrib_name_,
            policy_name_,
            FixedWidthString(time_avg_),
            FixedWidthString(time_max_)};
  }

  static std::string FixedWidthString(double d) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << d;
    return ss.str();
  }
};
}  // namespace amr