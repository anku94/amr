//
// Created by Ankush J on 7/4/23.
//

#include "common.h"
#include "globals.h"
#include "iterative/solver.h"
#include "lb_policies.h"

namespace amr {
int LoadBalancePolicies::AssignBlocksCppIter(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    void* opts) {
  int rv = 0;

  double max_iter_frac =
      Globals.config->GetParamOrDefault("cdpi_max_iter_frac", 0);
  int config_max_iters =
      Globals.config->GetParamOrDefault("cdpi_max_iters", 250);

  int nblocks = costlist.size();
  int max_iters = 0;

  if (max_iter_frac != 0) {
    max_iters = max_iter_frac * nblocks;
  } else {
    max_iters = config_max_iters;
  }

  logf(LOG_DBUG, "[CDPI] Max iters: %d", max_iters);

  double avg_cost, max_cost_lpt, max_cost_cpp, max_cost_iter;

  rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  if (rv) return rv;

  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_cpp);

  auto solver = Solver();
  int iters;

  solver.AssignBlocks(costlist, ranklist, nranks, max_iters);
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_iter);

  logf(LOG_DBUG, "IterativeSolver finished. Took %d iters.", iters);
  logf(LOG_DBUG,
       "Initial Cost: %.0lf, Target Cost: %.0lf.\n"
       "\t- Avg Cost: %.0lf, Max Cost: %.0lf",
       max_cost_cpp, max_cost_lpt, avg_cost, max_cost_iter);
  return 0;
}
}  // namespace amr
