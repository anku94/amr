//
// Created by Ankush J on 7/14/23.
//

#pragma once

#include "fort.hpp"
#include "lb_policies.h"
#include "policy.h"
#include "writable_file.h"

#include <pdlfs-common/env.h>

namespace amr {
class ScaleExecLog : public WritableCSVFile {
 public:
  ScaleExecLog(pdlfs::Env* const env, const std::string& fpath)
      : WritableCSVFile(env, fpath) {}

  void WriteRow(int nblocks, int nranks, double iter_time, double rt_avg,
                double rt_max) {
    char str[1024];
    int len = snprintf(str, 1024, "%d,%d,%.2f,%.2f,%.2f\n", nblocks, nranks,
                       iter_time, rt_avg, rt_max);
    assert(len < 1024);
    Append(str, len);
  }

 protected:
  void WriteHeader() override {
    const char* str = "nblocks,nranks,iter_time,rt_avg,rt_max\n";
    Append(str);
  }
};

#define LOG_PATH(x) \
  PolicyUtils::GetLogPath(opts_.output_dir, opts_.policy_name, x)

class ScaleExecCtx {
 public:
  explicit ScaleExecCtx(const PolicyExecOpts& opts)
      : opts_(opts), log_(opts_.env, LOG_PATH("exec.csv")) {}

  int AssignBlocks(int nranks, std::vector<double> const& cost_list,
                   fort::char_table& table) {
    int rv;
    std::vector<int> rank_list;
    std::vector<double> rank_times;

    uint64_t lb_beg = pdlfs::Env::NowMicros();
    for (int iter = 0; iter < kMaxIters; iter++) {
      rv = LoadBalancePolicies::AssignBlocks(opts_.lb_policy, cost_list,
                                             rank_list, nranks);
    }
    uint64_t lb_end = pdlfs::Env::NowMicros();

    double rtavg, rtmax;
    PolicyUtils::ComputePolicyCosts(nranks, cost_list, rank_list, rank_times,
                                    rtavg, rtmax);

    double iter_time = (lb_end - lb_beg) / kMaxIters;

    log_.WriteRow(cost_list.size(), nranks, iter_time, rtavg, rtmax);
    table << cost_list.size() << nranks << iter_time << rtavg << rtmax
          << fort::endr;

    return rv;
  }

  static void LogHeader(fort::char_table& table) {
    table << fort::header << "Num Blocks"
          << "Num Ranks"
          << "Iter Time"
          << "Rank Time (avg)"
          << "Rank Time (max)" << fort::endr;
  }

 private:
  PolicyExecOpts const opts_;
  static constexpr int kMaxIters = 1;
  ScaleExecLog log_;
};
}  // namespace amr