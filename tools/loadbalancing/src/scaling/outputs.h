//
// Created by Ankush J on 7/18/23.
//

#pragma once

#include "policy.h"
#include "writable_file.h"

#include <pdlfs-common/env.h>

namespace amr {
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

    table_ << std::fixed << std::setprecision(0) << policy_name_cleaned
           << nblocks << nranks << iter_time << rt_avg << rt_max
           << std::setprecision(2) << loc_cost << fort::endr;
  }

  std::string GetTabularStr() const { return table_.to_string(); }

 private:
  void WriteHeader() override {
    const char* str =
        "policy,nblocks,nranks,iter_time,rt_avg,rt_max,loc_cost\n";
    Append(str);

    table_ << fort::header << "Policy"
           << "Num Blocks"
           << "Num Ranks"
           << "Iter Time (us)"
           << "Rank Time (avg)"
           << "Rank Time (max)"
           << "Loc Cost (%)" << fort::endr;
  }

 private:
  fort::char_table table_;
};
}  // namespace amr