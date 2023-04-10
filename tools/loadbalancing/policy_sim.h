#pragma once

#include "lb_policies.h"
#include "prof_set_reader.h"

#include <vector>

#define PROF_DIR "/mnt/ltio/parthenon-topo/profile20/"

namespace amr {
class PolicySim {
 public:
  PolicySim() : excess_cost_(0), total_cost_avg_(0), total_cost_max_(0) {}

  void Run() {
    ProfSetReader psr;
    psr.AddProfile(PROF_DIR "prof.merged.evt0.csv");
    psr.AddProfile(PROF_DIR "prof.merged.evt1.csv");

    int nread;
    int nts = 0;

    std::vector<int> times;

    while ((nread = psr.ReadTimestep(times)) > 0) {
      CheckAssignment(times);

      nts++;
      if (nts % 100 == 0) {
        printf("\rTS Read: %d", nts);
        // break;
      }
    }

    printf("\n");
    logf(LOG_INFO, "Excess Cost: \t%.7f s", excess_cost_ / 1e6);
    logf(LOG_INFO, "Avg Cost: \t%.7f s", total_cost_avg_ / 1e6);
    logf(LOG_INFO, "Max Cost: \t%.7f s", total_cost_max_ / 1e6);
  }

  void CheckAssignment(std::vector<int>& times) {
    int nranks = 512;
    int nblocks = times.size();
    std::vector<int> rank_times(nranks, 0);

    std::vector<double> costlist(nblocks, 1.0f);
    std::vector<int> ranklist(nblocks, -1);

    // Policies::AssignBlocksRoundRobin(costlist, ranklist, nranks);
    Policies::AssignBlocksContiguous(costlist, ranklist, nranks);

    for (int bid = 0; bid < nblocks; bid++) {
      int block_rank = ranklist[bid];
      int block_cost = times[bid];
      rank_times[block_rank] += block_cost;
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
  }

 private:
  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;
};
}  // namespace amr
