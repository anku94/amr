//
// Created by Ankush J on 4/12/23.
//

#pragma once

#include "fort.hpp"
#include "writable_file.h"

#include <pdlfs-common/env.h>

namespace amr {
// fwd decl
class PolicyExecCtx;

class PolicyStats {
 public:
  PolicyStats()
      : ts_(0),
        excess_cost_(0),
        total_cost_avg_(0),
        total_cost_max_(0),
        locality_score_sum_(0) {}

  void LogTimestep(PolicyExecCtx* pctx, int nranks,
                   std::vector<double> const& cost_actual,
                   std::vector<int> const& rank_list);

  static void LogHeader(fort::char_table& table);

  void LogSummary(fort::char_table& table) const;

  static std::string FormatProp(double prop, const char* suffix) {
    char buf[1024];
    snprintf(buf, 1024, "%.1f %s", prop, suffix);
    return {buf};
  }

 private:
  void WriteSummary(WritableFile& fd, double avg, double max) {
    if (ts_ == 0) {
      const char* header = "ts,avg_us,max_us\n";
      fd.Append(header);
    }

    char buf[1024];
    int buf_len = snprintf(buf, 1024, " %d,%.0lf,%.0lf\n", ts_, avg, max);
    fd.Append(std::string(buf, buf_len));
  }

  void WriteDetailed(WritableFile& fd, std::vector<double> const& cost_actual,
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

  void WriteRankSums(WritableFile& fd, std::vector<double>& rank_times) {
    int nranks = rank_times.size();
    if (ts_ == 0) {
      fd.Append(reinterpret_cast<const char*>(&nranks), sizeof(int));
    }
    fd.Append(reinterpret_cast<const char*>(rank_times.data()),
              sizeof(double) * nranks);
  }

  //
  // Use an arbitary model to compute
  // Intuition: amount of linear locality captured (lower is better)
  // cost of 1 for neighboring ranks
  // cost of 2 for same node (hardcoded rn)
  // cost of 3 for arbitrary communication
  //
  static double ComputeLocScore(std::vector<int> const& rank_list) {
    int nb = rank_list.size();
    int local_score = 0;

    for (int bidx = 0; bidx < nb - 1; bidx++) {
      int p = rank_list[bidx];
      int q = rank_list[bidx + 1];

      // Nodes for p and q, computed using assumptions
      int pn = p / 16;
      int qn = q / 16;

      if (p == q) {
        // nothing
      } else if (abs(q - p) == 1) {
        local_score += 1;
      } else if (qn == pn) {
        local_score += 2;
      } else {
        local_score += 3;
      }
    }

    double norm_score = local_score * 1.0 / nb;
    return norm_score;
  }

  int ts_;

  // cost is assumed to be us
  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;

  double locality_score_sum_;
};
}  // namespace amr
