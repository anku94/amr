#include "lb_hybrid2.h"

#include <cassert>
#include <numeric>

#include "common.h"
#include "lb_policies.h"
#include "policy_utils.h"
#include "policy_wopts.h"

struct PartialLPTSolution {
  // inputs
  std::vector<double> const& costlist;
  std::vector<int> ranklist;
  std::vector<double> const& rank_costs;
  const int nlpt;  // max rank count for LPT
  // outputs
  std::vector<int> lpt_ranks;
  std::vector<int> lpt_blocks;
  std::vector<int> ranklist_lpt;
  double cost_avg;
  double cost_max;

  PartialLPTSolution(std::vector<double> const& costlist,
                     std::vector<int> const& ranklist,
                     std::vector<double> const& rank_costs, int max_nlpt)
      : costlist(costlist),
        ranklist(ranklist),
        rank_costs(rank_costs),
        nlpt(max_nlpt),
        cost_avg(0),
        cost_max(0) {}
};

void PopulateRanklistWithSolution(std::vector<int>& ranklist,
                                  PartialLPTSolution const& solution) {
  int lpt_nblocks = solution.lpt_blocks.size();

  for (int lpt_bid = 0; lpt_bid < lpt_nblocks; lpt_bid++) {
    int lpt_rid = solution.ranklist_lpt[lpt_bid];
    int real_bid = solution.lpt_blocks[lpt_bid];
    int real_rid = solution.lpt_ranks[lpt_rid];

    ranklist[real_bid] = real_rid;
  }
}

static void ComputePartialLPTSolution(PartialLPTSolution& solution) {
  int rv = 0;
  solution.lpt_ranks = amr::HybridAssignmentCppFirst::GetLPTRanksV3(
      solution.costlist, solution.ranklist, solution.rank_costs, solution.nlpt);
  solution.lpt_blocks = amr::HybridAssignmentCppFirst::GetBlocksForRanks(
      solution.ranklist, solution.lpt_ranks);

  std::vector<double> costlist_lpt;
  std::vector<int> ranklist_lpt;

  for (auto bid : solution.lpt_blocks) {
    costlist_lpt.push_back(solution.costlist[bid]);
  }

  rv = amr::LoadBalancePolicies::AssignBlocks(
      amr::HybridAssignmentCppFirst::kLPTPolicyStr, costlist_lpt,
      solution.ranklist_lpt, solution.lpt_ranks.size());

  if (rv) {
    ABORT("[HybridCppFirst] LPT failed");
  }

  int lpt_nblocks = solution.lpt_blocks.size();

  for (int lpt_bid = 0; lpt_bid < lpt_nblocks; lpt_bid++) {
    int lpt_rid = solution.ranklist_lpt[lpt_bid];
    int real_bid = solution.lpt_blocks[lpt_bid];
    int real_rid = solution.lpt_ranks[lpt_rid];

    solution.ranklist[real_bid] = real_rid;
  }

  int nranks = solution.rank_costs.size();
  std::vector<double> rank_times(nranks, 0);
  amr::PolicyUtils::ComputePolicyCosts(nranks, solution.costlist,
                                       solution.ranklist, rank_times,
                                       solution.cost_avg, solution.cost_max);
}

namespace amr {
int LoadBalancePolicies::AssignBlocksHybridCppFirst(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    PolicyOptsHybridCDPFirst const& opts) {
  int rv = 0;

  bool v2 = opts.v2;
  double lpt_frac = opts.lpt_frac;
  int lpt_ranks = nranks * lpt_frac;
  int alt_solncnt_max = opts.alt_solncnt_max;

  static bool first_time = true;

  if (first_time) {
    logv(__LOG_ARGS__, LOG_INFO,
         "[HybridCppFirst] LPT ranks: %d, V2: %s, altcnt: %d, CDP: %s",
         lpt_ranks, v2 ? "yes" : "no", alt_solncnt_max,
         HybridAssignmentCppFirst::kCDPPolicyStr);
    first_time = false;
  }

  auto hacf = HybridAssignmentCppFirst(lpt_ranks, alt_solncnt_max);

  if (v2) {
    rv = hacf.AssignBlocksV2(costlist, ranklist, nranks, MPI_COMM_NULL);
  } else {
    rv = hacf.AssignBlocks(costlist, ranklist, nranks);
  }

  return rv;
}

int LoadBalancePolicies::AssignBlocksParallelHybridCDPFirst(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    PolicyOptsHybridCDPFirst const& opts, MPI_Comm comm, int mympirank,
    int nmpiranks) {
  int rv = 0;

  bool v2 = opts.v2;
  double lpt_frac = opts.lpt_frac;
  int lpt_ranks = nranks * lpt_frac;
  int alt_solncnt_max = opts.alt_solncnt_max;

  static bool first_time = true;

  if (first_time) {
    logv(__LOG_ARGS__, LOG_INFO,
         "[HybridCppFirst] LPT ranks: %d, V2: %s, altcnt: %d, CDP: %s",
         lpt_ranks, v2 ? "yes" : "no", alt_solncnt_max,
         HybridAssignmentCppFirst::kCDPPolicyStr);
    first_time = false;
  }

  auto hacf = HybridAssignmentCppFirst(lpt_ranks, alt_solncnt_max);

  if (v2) {
    rv = hacf.AssignBlocksV2(costlist, ranklist, nranks, comm);
  } else {
    rv = hacf.AssignBlocks(costlist, ranklist, nranks);
  }

  return rv;
}

int HybridAssignmentCppFirst::AssignBlocks(std::vector<double> const& costlist,
                                           std::vector<int>& ranklist,
                                           int nranks) {
  nblocks_ = costlist.size();
  nranks_ = nranks;

  std::vector<double> rank_times;
  double rank_time_max, rank_time_avg;

  int rv = LoadBalancePolicies::AssignBlocks(kCDPPolicyStr, costlist, ranklist,
                                             nranks);

  PolicyUtils::ComputePolicyCosts(nranks, costlist, ranklist, rank_times,
                                  rank_time_avg, rank_time_max);

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirst] Costs after CPP_Iter, avg: %.0lf, max: %.0lf",
       rank_time_avg, rank_time_max);

  rank_costs_.clear();
  rank_costs_.resize(nranks, 0.0);

  for (size_t i = 0; i < costlist.size(); i++) {
    rank_costs_[ranklist[i]] += costlist[i];
  }

  auto lpt_ranks = GetLPTRanks(costlist, ranklist);
  auto lpt_blocks = GetBlocksForRanks(ranklist, lpt_ranks);
  int lpt_nblocks = lpt_blocks.size();
  int lpt_nranks = lpt_ranks.size();

  std::vector<double> costlist_lpt;
  std::vector<int> ranklist_lpt;

  for (auto bid : lpt_blocks) {
    costlist_lpt.push_back(costlist[bid]);
  }

  rv = LoadBalancePolicies::AssignBlocks(kLPTPolicyStr, costlist_lpt,
                                         ranklist_lpt, lpt_nranks);

  if (rv) {
    ABORT("[HybridCppFirst] LPT failed");
  }

  for (int lpt_bid = 0; lpt_bid < lpt_nblocks; lpt_bid++) {
    int lpt_rid = ranklist_lpt[lpt_bid];
    int real_bid = lpt_blocks[lpt_bid];
    int real_rid = lpt_ranks[lpt_rid];

    ranklist[real_bid] = real_rid;
  }

  PolicyUtils::ComputePolicyCosts(nranks, costlist, ranklist, rank_times,
                                  rank_time_avg, rank_time_max);
  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirst] Costs after LPT, avg: %.0lf, max: %.0lf",
       rank_time_avg, rank_time_max);

  return rv;
}

int HybridAssignmentCppFirst::AssignBlocksV2(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    MPI_Comm comm) {
  int rv = 0;

  nblocks_ = costlist.size();
  nranks_ = nranks;

  std::vector<double> rank_times;
  double rank_time_max, rank_time_avg;

  if (comm == MPI_COMM_NULL) {
    rv = LoadBalancePolicies::AssignBlocks(kCDPPolicyStr, costlist, ranklist,
                                           nranks);
  } else {
    rv = LoadBalancePolicies::AssignBlocksParallel(kParCDPPolicyStr, costlist,
                                                   ranklist, nranks, comm);
  }

  PolicyUtils::ComputePolicyCosts(nranks, costlist, ranklist, rank_times,
                                  rank_time_avg, rank_time_max);

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirst] Costs after CPP_Iter, avg: %.0lf, max: %.0lf",
       rank_time_avg, rank_time_max);

  rank_costs_.clear();
  rank_costs_.resize(nranks, 0.0);

  for (size_t i = 0; i < costlist.size(); i++) {
    rank_costs_[ranklist[i]] += costlist[i];
  }

  PartialLPTSolution solution(costlist, ranklist, rank_costs_, lpt_rank_count_);
  ComputePartialLPTSolution(solution);
  auto ranklist_best = solution.ranklist;

  // Explore alternate solutions with lower locality loss
  // than the given LPT parameter
  int nlpt_alt = solution.lpt_ranks.size();

  for (int alt_idx = 0; alt_idx < alt_max_; alt_idx++) {
    PartialLPTSolution alt(costlist, ranklist, rank_costs_, nlpt_alt / 2);
    ComputePartialLPTSolution(alt);

    logv(__LOG_ARGS__, LOG_DBUG, "Cost main: %.0lf, alt: %.0lf (%d vs %d)",
         solution.cost_max, alt.cost_max, solution.lpt_ranks.size(),
         alt.lpt_ranks.size());

    if (alt.cost_max <= solution.cost_max) {
      ranklist_best = alt.ranklist;
      nlpt_alt = nlpt_alt / 2;
    } else {
      break;
    }
  }

  ranklist = ranklist_best;
  return rv;
}

std::vector<int> HybridAssignmentCppFirst::GetLPTRanks(
    std::vector<double> const& costlist,
    std::vector<int> const& ranklist) const {
  std::vector<int> lpt_ranks;  // ranks that would benefit from LPT
  std::vector<std::pair<double, int>> cost_ranks;  // costs + idxes to sort

  for (int i = 0; i < nranks_; i++) {
    cost_ranks.push_back(std::make_pair(rank_costs_[i], i));
  }

  std::sort(
      cost_ranks.begin(), cost_ranks.end(),
      [](std::pair<double, int> const& a, std::pair<double, int> const& b) {
        return a.first > b.first;
      });  // sort in descending order

  double cost_sum = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_avg = cost_sum / nranks_;

  // we are trying to find ranks to run LPT for
  // this includes k largest ranks, + a number of underloaded ranks
  // such that the average of all of these is under the total cost
  double cost_sum_lpt = 0.0;
  double cost_avg_lpt = 0;
  int ranks_for_lpt = lpt_rank_count_;

  for (int i = 0; i < lpt_rank_count_; i++) {
    cost_sum_lpt += cost_ranks[i].first;
    lpt_ranks.push_back(cost_ranks[i].second);
  }

  cost_avg_lpt = cost_sum_lpt / ranks_for_lpt;

  for (int i = nranks_ - 1; i >= lpt_rank_count_; i--) {
    logv(__LOG_ARGS__, LOG_DBUG, "Cur rank: %d, cost: %.0lf", i,
         cost_ranks[i].first);

    double cost_sum_with_rank = cost_sum_lpt + cost_ranks[i].first;
    double cost_avg_with_rank = cost_sum_with_rank / (ranks_for_lpt + 1);

#define DOUBLE_EPSILON 1e-3

    if (cost_avg_with_rank > cost_avg + DOUBLE_EPSILON) {
      cost_sum_lpt = cost_sum_with_rank;
      cost_avg_lpt = cost_avg_with_rank;
      lpt_ranks.push_back(cost_ranks[i].second);
      ranks_for_lpt++;
    } else {
      break;
    }
  }

  assert(lpt_ranks.size() <= nranks_);

  logv(__LOG_ARGS__, LOG_DBG2, "Selected for LPT: %d initial, %d rest",
       lpt_rank_count_, ranks_for_lpt - lpt_rank_count_);
  logv(__LOG_ARGS__, LOG_DBG2, "Cost avg: %.2f, cost avg LPT: %.2f", cost_avg,
       cost_avg_lpt);

  return lpt_ranks;
}

std::vector<int> HybridAssignmentCppFirst::GetLPTRanksV2(
    std::vector<double> const& costlist,
    std::vector<int> const& ranklist) const {
  std::vector<std::pair<double, int>> cost_ranks;  // costs + idxes to sort

  for (int i = 0; i < nranks_; i++) {
    cost_ranks.push_back(std::make_pair(rank_costs_[i], i));
  }

  std::sort(
      cost_ranks.begin(), cost_ranks.end(),
      [](std::pair<double, int> const& a, std::pair<double, int> const& b) {
        return a.first > b.first;
      });  // sort in descending order

  double cost_sum = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_avg = cost_sum / nranks_;

  // we are trying to find ranks to run LPT for
  // this includes k largest ranks, + a number of underloaded ranks
  // such that the average of all of these is under the total cost
  double cost_sum_lpt = 0.0;
  double cost_avg_lpt = 0;
  int ranks_for_lpt = lpt_rank_count_;

  for (int i = 0; i < lpt_rank_count_; i++) {
    cost_sum_lpt += cost_ranks[i].first;
  }

  cost_avg_lpt = cost_sum_lpt / ranks_for_lpt;

  int incl_ranks_front = lpt_rank_count_;
  int incl_ranks_back = 0;

  for (int i = nranks_ - 1; i >= lpt_rank_count_; i--) {
    logv(__LOG_ARGS__, LOG_DBUG, "Cur rank: %d, cost: %.0lf", i,
         cost_ranks[i].first);

    double cost_toadd = cost_ranks[i].first;
    double cost_todel = cost_ranks[incl_ranks_front - 1].first;

    double cost_sum_new = cost_sum_lpt + cost_toadd - cost_todel;
    double cost_avg_new = cost_sum_new / ranks_for_lpt;

#define DOUBLE_EPSILON 1e-3

    logv(__LOG_ARGS__, LOG_DBUG, "[HybridCppFirstV2] avg_new: %.2f, avg: %.2f",
         cost_avg_new, cost_avg);

    cost_sum_lpt = cost_sum_new;
    cost_avg_lpt = cost_avg_new;
    incl_ranks_back++;
    incl_ranks_front--;

    if (cost_avg_new > cost_avg + DOUBLE_EPSILON) {
      if (incl_ranks_front == 0) {
        logv(__LOG_ARGS__, LOG_WARN,
             "Can't LPT-rebalance with given constraints!");
        return std::vector<int>();
      }
    } else {
      break;
    }
  }

  assert(incl_ranks_front + incl_ranks_back == lpt_rank_count_);
  assert(incl_ranks_front + incl_ranks_back <= nranks_);

  std::vector<int> lpt_ranks;  // ranks that would benefit from LPT
  for (int i = 0; i < incl_ranks_front; i++) {
    lpt_ranks.push_back(cost_ranks[i].second);
  }

  for (int i = nranks_ - incl_ranks_back; i < nranks_; i++) {
    lpt_ranks.push_back(cost_ranks[i].second);
  }

  assert(lpt_ranks.size() <= nranks_);

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirstV2] Selected for LPT: %d initial, %d rest"
       "(Total: %d)",
       incl_ranks_front, incl_ranks_back, lpt_ranks.size());

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirstV2] Cost avg: %.2f, cost avg LPT: %.2f", cost_avg,
       cost_avg_lpt);

  return lpt_ranks;
}

std::vector<int> HybridAssignmentCppFirst::GetLPTRanksV3(
    std::vector<double> const& costlist, std::vector<int> const& ranklist,
    std::vector<double> const& rank_costs, const int nmax) {
  int nranks = rank_costs.size();

  std::vector<std::pair<double, int>> cost_ranks;  // costs + idxes to sort

  for (int i = 0; i < nranks; i++) {
    cost_ranks.push_back(std::make_pair(rank_costs[i], i));
  }

  std::sort(
      cost_ranks.begin(), cost_ranks.end(),
      [](std::pair<double, int> const& a, std::pair<double, int> const& b) {
        return a.first > b.first;
      });  // sort in descending order

  double cost_sum = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_avg = cost_sum / nranks;

  double cost_sum_lpt = cost_ranks[0].first;
  double cost_avg_lpt = cost_ranks[0].first;

  int incl_ranks_front = 1;
  int incl_ranks_back = 0;

  while (incl_ranks_front + incl_ranks_back < nmax) {
    double cost_front = cost_ranks[incl_ranks_front].first;
    double cost_back = cost_ranks[nranks - 1 - incl_ranks_back].first;

    double diff_avg_front = cost_front - cost_avg;
    double diff_avg_back = cost_avg - cost_back;

    if (cost_avg_lpt > cost_avg) {
      // grow from the back
      cost_sum_lpt += cost_back;
      incl_ranks_back++;
      cost_avg_lpt = cost_sum_lpt / (incl_ranks_front + incl_ranks_back);
    } else if (cost_avg_lpt < cost_avg) {
      // grow from the front
      cost_sum_lpt += cost_front;
      incl_ranks_front++;
      cost_avg_lpt = cost_sum_lpt / (incl_ranks_front + incl_ranks_back);
    } else {
      break;
    }

    logv(__LOG_ARGS__, LOG_DBG2, "Cost front: %.2f, back: %.2f, avg: %.2f",
         cost_front, cost_back, cost_avg);

    double diff_threshold = cost_avg * kLPTv3DiffThreshold;

    logv(__LOG_ARGS__, LOG_DBG2,
         "Diff front: %.2f, back: %.2f, threshold: %.2f", diff_avg_front,
         diff_avg_back, diff_threshold);

    if (diff_avg_front < diff_threshold && diff_avg_back < diff_threshold) {
      logv(__LOG_ARGS__, LOG_DBUG,
           "Breaking prematurely as remaining ranks are balanced");
      break;
    }
  }

  assert(incl_ranks_front + incl_ranks_back <= nmax);

  std::vector<int> lpt_ranks;  // ranks that would benefit from LPT
  for (int i = 0; i < incl_ranks_front; i++) {
    lpt_ranks.push_back(cost_ranks[i].second);
  }

  for (int i = nranks - incl_ranks_back; i < nranks; i++) {
    lpt_ranks.push_back(cost_ranks[i].second);
  }

  assert(lpt_ranks.size() <= nranks);

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirstV2] Selected for LPT: %d initial, %d rest"
       "(Total: %d/%d)",
       incl_ranks_front, incl_ranks_back, (int)lpt_ranks.size(), nmax);

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridCppFirstV2] Cost avg: %.2f, cost avg LPT: %.2f", cost_avg,
       cost_avg_lpt);

  return lpt_ranks;
}

std::vector<int> HybridAssignmentCppFirst::GetBlocksForRanks(
    std::vector<int> const& ranklist, std::vector<int> const& selected_ranks) {
  std::vector<int> selected_bids;  // block ids
  int max_rank = *std::max_element(ranklist.begin(), ranklist.end());
  std::vector<bool> selected_ranks_map(max_rank + 1, false);

  for (auto r : selected_ranks) {
    selected_ranks_map[r] = true;
  }

  for (size_t i = 0; i < ranklist.size(); i++) {
    if (selected_ranks_map[ranklist[i]]) {
      logv(__LOG_ARGS__, LOG_DBG3, "Block %d, rank %d", i, ranklist[i]);
      selected_bids.push_back(i);
    }
  }

  return selected_bids;
}
}  // namespace amr
