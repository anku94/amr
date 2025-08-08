//
// Created by Ankush J on 7/20/23.
//

#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace amr {
class PolicyTools {
 public:
  static double GetDisorder(std::vector<int>& array) {
    double disorder = 0;
    for (int i = 0; i < array.size() - 1; i++) {
      disorder += std::fabs(array[i + 1] - array[i]);
    }

    return disorder;
  }

  static double GetLinearDisorder(std::vector<int>& array) {
    double loc_score = 0;
    for (int i = 0; i < array.size(); i++) {
      loc_score += i * array[i];
    }

    return loc_score;
  }

  static void ComputePolicyCosts(int nranks,
                                 std::vector<double> const& cost_list,
                                 std::vector<int> const& rank_list,
                                 std::vector<double>& rank_times,
                                 double& rank_time_avg, double& rank_time_max) {
    rank_times.resize(nranks, 0);
    int nblocks = cost_list.size();

    for (int bid = 0; bid < nblocks; bid++) {
      int block_rank = rank_list[bid];
      rank_times[block_rank] += cost_list[bid];
    }

    int const& (*max_func)(int const&, int const&) = std::max<int>;
    rank_time_max = std::accumulate(rank_times.begin(), rank_times.end(),
                                    rank_times.front(), max_func);
    uint64_t rtsum =
        std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
    rank_time_avg = rtsum * 1.0 / nranks;
  }

  static void ComputePolicyCosts(int nranks,
                                 std::vector<double> const& cost_list,
                                 std::vector<int> const& rank_list,
                                 double& rank_time_avg, double& rank_time_max) {
    std::vector<double> rank_times;
    ComputePolicyCosts(nranks, cost_list, rank_list, rank_times, rank_time_avg,
                       rank_time_max);
  }
};
}  // namespace amr
