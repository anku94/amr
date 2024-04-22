//
// Created by Ankush J on 7/4/23.
//
#include "common.h"
#include "iterative/solver.h"
#include "lb_policies.h"
#include "policy_wopts.h"

namespace amr {
int LoadBalancePolicies::AssignBlocksCppIter(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    PolicyOptsCDPI const& opts) {
  int rv = 0;

  double max_iter_frac = opts.niter_frac;
  int config_max_iters = opts.niters;

  int nblocks = costlist.size();
  int max_iters = 0;

  if (max_iter_frac != 0) {
    max_iters = max_iter_frac * nblocks;
  } else {
    max_iters = config_max_iters;
  }

  logv(__LOG_ARGS__, LOG_DBUG, "[CDPI] Max iters: %d", max_iters);

  double avg_cost, max_cost_lpt, max_cost_cpp, max_cost_iter;

  rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  if (rv) return rv;

  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_cpp);

  auto solver = Solver();
  int iters;

  solver.AssignBlocks(costlist, ranklist, nranks, max_iters);
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_iter);

  logv(__LOG_ARGS__, LOG_DBUG, "IterativeSolver finished. Took %d iters.",
       iters);
  logv(__LOG_ARGS__, LOG_DBUG,
       "Initial Cost: %.0lf, Target Cost: %.0lf.\n"
       "\t- Avg Cost: %.0lf, Max Cost: %.0lf",
       max_cost_cpp, max_cost_lpt, avg_cost, max_cost_iter);
  return 0;
}
}  // namespace amr
