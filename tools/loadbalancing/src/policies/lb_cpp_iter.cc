//
// Created by Ankush J on 7/4/23.
//

#include "common.h"
#include "iterative/solver.h"
#include "lb_policies.h"

namespace amr {
int LoadBalancePolicies::AssignBlocksCppIter(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    void* opts) {
  int rv = AssignBlocksLPT(costlist, ranklist, nranks);
  if (rv) return rv;

  int max_iters = Solver::kMaxIters;

  // if (opts != nullptr) {
    // max_iters = *(int*)opts;
  // }

  double avg_cost, max_cost_lpt, max_cost_cpp, max_cost_iter;
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_lpt);
  logf(LOG_DBUG, "LPT. Avg Cost: %.0lf, Max Cost: %.0lf\n", avg_cost,
       max_cost_lpt);

  rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  if (rv) return rv;

  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_cpp);
  // logf(LOG_INFO, "Contig++. Avg Cost: %.0lf, Max Cost: %.0lf\n", avg_cost,
       // max_cost_cpp);

  auto solver = Solver();
  int iters;
  solver.AssignBlocks(costlist, ranklist, nranks, max_cost_lpt, iters, max_iters);
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_iter);

  logf(LOG_DBUG, "IterativeSolver finished. Took %d iters.", iters);
  logf(LOG_DBUG,
       "Initial Cost: %.0lf, Target Cost: %.0lf.\n"
       "\t- Avg Cost: %.0lf, Max Cost: %.0lf",
       max_cost_cpp, max_cost_lpt, avg_cost, max_cost_iter);
  return 0;
}
}  // namespace amr
