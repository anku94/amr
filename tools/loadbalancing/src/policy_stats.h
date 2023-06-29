//
// Created by Ankush J on 4/12/23.
//

#pragma once

#include "fort.hpp"

#include <pdlfs-common/env.h>

#define SAFE_IO(func, msg) \
  s = func;                \
  if (!s.ok()) {           \
    ABORT(msg);            \
  }

namespace amr {
class PolicyStats {
 public:
  PolicyStats()
      : ts_(0),
        excess_cost_(0),
        total_cost_avg_(0),
        total_cost_max_(0),
        locality_score_sum_(0) {}

  void LogTimestep(int nranks, pdlfs::WritableFile* fd,
                   std::vector<double> const& cost_actual,
                   std::vector<int> const& rank_list) {
    int nblocks = cost_actual.size();
    std::vector<double> rank_times(nranks, 0);

    for (int bid = 0; bid < nblocks; bid++) {
      int block_rank = rank_list[bid];
      rank_times[block_rank] += cost_actual[bid];
    }

    int const& (*max_func)(int const&, int const&) = std::max<int>;
    int rtmax = std::accumulate(rank_times.begin(), rank_times.end(),
                                rank_times.front(), max_func);
    uint64_t rtsum =
        std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
    double rtavg = rtsum * 1.0 / nranks;

    excess_cost_ += (rtmax - rtavg);
    total_cost_avg_ += rtavg;
    total_cost_max_ += rtmax;
    locality_score_sum_ += ComputeLocScore(rank_list);

    if (ts_ == 0) {
      WriteHeader(fd);
    }
    WriteData(fd, ts_, rtavg, rtmax);

    ts_++;
  }

  void LogTimestepVerbose(int nranks, pdlfs::WritableFile* fd,
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

    pdlfs::Status s;
    SAFE_IO(fd->Append(ss.str()), "Write failed");
  }

  static void LogHeader(fort::char_table& table);

  void LogSummary(fort::char_table& table) const;

  static std::string FormatProp(double prop, const char* suffix) {
    char buf[1024];
    snprintf(buf, 1024, "%.1f %s", prop, suffix);
    return {buf};
  }

 private:
  static void WriteHeader(pdlfs::WritableFile* fd) {
    const char* header = "ts,avg_us,max_us\n";
    pdlfs::Status s;
    SAFE_IO(fd->Append(header), "Write failed");
  }

  static void WriteData(pdlfs::WritableFile* fd, int ts, double avg,
                        double max) {
    char buf[1024];
    int buf_len = snprintf(buf, 1024, " %d,%.0lf,%.0lf\n", ts, avg, max);
    pdlfs::Status s;
    SAFE_IO(fd->Append(pdlfs::Slice(buf, buf_len)), "Write failed");
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
