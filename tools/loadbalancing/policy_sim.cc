//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

namespace amr {
void PolicySim::SimulateTrace() {
  std::vector<std::string> files = LocateRelevantFiles(options_.prof_dir);

  if (files.empty()) {
    ABORT("no trace files found!");
  }

  ProfSetReader psr;
  for (auto& f: files) {
    std::string full_path = std::string(options_.prof_dir) + "/" + f;
    logf(LOG_DBUG, "+ Adding file: %s", full_path.c_str());
    psr.AddProfile(full_path);
  }

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
  logf(LOG_INFO, "Excess Cost: \t%.2f s", excess_cost_ / 1e6);
  logf(LOG_INFO, "Avg Cost: \t%.2f s", total_cost_avg_ / 1e6);
  logf(LOG_INFO, "Max Cost: \t%.2f s", total_cost_max_ / 1e6);
}

void PolicySim::CheckAssignment(std::vector<int>& times) {
  int nranks = 512;
  int nblocks = times.size();
  std::vector<int> rank_times(nranks, 0);

  std::vector<double> costlist(nblocks, 1.0f);
  std::vector<int> ranklist(nblocks, -1);

  Policies::AssignBlocks(options_.policy, costlist, ranklist, nranks);

  for (int bid = 0; bid < nblocks; bid++) {
    int block_rank = ranklist[bid];
    int block_cost = times[bid];
    rank_times[block_rank] += block_cost;
  }

  int const& (*max_func)(int const&, int const&) = std::max<int>;
  int rtmax = std::accumulate(rank_times.begin(), rank_times.end(),
                              rank_times.front(), max_func);
  uint64_t rtsum = std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
  double rtavg = rtsum * 1.0 / nranks;

  excess_cost_ += (rtmax - rtavg);
  total_cost_avg_ += rtavg;
  total_cost_max_ += rtmax;
}
}  // namespace amr