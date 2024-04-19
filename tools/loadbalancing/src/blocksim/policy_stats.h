//
// Created by Ankush J on 4/12/23.
//

#pragma once

#include "fort.hpp"
#include "policy.h"
#include "writable_file.h"

#include <pdlfs-common/env.h>

namespace amr {
// fwd decl
class PolicyExecCtx;

#define LOG_PATH(x) PolicyUtils::GetLogPath(opts.output_dir, opts.policy_id, x)

class PolicyStats {
 public:
  PolicyStats(PolicyExecOpts& opts)
      : opts_(opts),
        ts_(0),
        excess_cost_(0),
        total_cost_avg_(0),
        total_cost_max_(0),
        locality_score_sum_(0),
        exec_time_us_(0),
        fd_summ_(opts.env, LOG_PATH("summ")),
        fd_det_(opts.env, LOG_PATH("det")),
        fd_ranksum_(opts.env, LOG_PATH("ranksum")) {}

  void LogTimestep(std::vector<double> const& cost_actual,
                   std::vector<int> const& rank_list, double exec_time_ts);

  static void LogHeader(fort::char_table& table);

  void LogSummary(fort::char_table& table) const;

  static std::string FormatProp(double prop, const char* suffix) {
    char buf[1024];
    snprintf(buf, 1024, "%.1f %s", prop, suffix);
    return {buf};
  }

 private:
  void WriteSummary(WritableFile& fd, double avg, double max) const {
    if (ts_ == 0) {
      const char* header = "ts,avg_us,max_us\n";
      fd.Append(header);
    }

    char buf[1024];
    int buf_len = snprintf(buf, 1024, " %d,%.0lf,%.0lf\n", ts_, avg, max);
    fd.Append(std::string(buf, buf_len));
  }

  static void WriteDetailed(WritableFile& fd,
                            std::vector<double> const& cost_actual,
                            std::vector<int> const& rank_list) {
    std::stringstream ss;
    for (auto c : cost_actual) {
      ss << (int)c << ",";
    }
    ss << std::endl;

    for (auto r : rank_list) {
      ss << r << ",";
    }
    ss << std::endl;

    fd.Append(ss.str());
  }

  void WriteRankSums(WritableFile& fd, std::vector<double>& rank_times) const {
    int nranks = rank_times.size();
    if (ts_ == 0) {
      fd.Append(reinterpret_cast<const char*>(&nranks), sizeof(int));
    }
    fd.Append(reinterpret_cast<const char*>(rank_times.data()),
              sizeof(double) * nranks);
  }

  const PolicyExecOpts opts_;

  int ts_;

  // cost is assumed to be us
  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;

  double locality_score_sum_;

  double exec_time_us_;

  WritableFile fd_summ_;
  WritableFile fd_det_;
  WritableFile fd_ranksum_;
};
}  // namespace amr
