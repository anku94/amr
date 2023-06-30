//
// Created by Ankush J on 5/4/23.
//

#include "policy_exec_ctx.h"

#include "fort.hpp"
#include "policy.h"

namespace amr {
void PolicyStats::LogTimestep(PolicyExecCtx* pctx, int nranks,
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
  uint64_t rtsum = std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
  double rtavg = rtsum * 1.0 / nranks;

  excess_cost_ += (rtmax - rtavg);
  total_cost_avg_ += rtavg;
  total_cost_max_ += rtmax;
  locality_score_sum_ += ComputeLocScore(rank_list);

  WriteSummary(pctx->fd_summ_, rtavg, rtmax);
  WriteDetailed(pctx->fd_det_, cost_actual, rank_list);
  WriteRankSums(pctx->fd_ranksum_, rank_times);

  ts_++;
}

void PolicyStats::LogHeader(fort::char_table& table) {
  table << "ExcessCost"
        << "AvgCost"
        << "MaxCost"
        << "LocScore";
}

void PolicyStats::LogSummary(fort::char_table& table) const {
  table << FormatProp(excess_cost_ / 1e6, "s")
        << FormatProp(total_cost_avg_ / 1e6, "s")
        << FormatProp(total_cost_max_ / 1e6, "s")
        << FormatProp(locality_score_sum_ * 100 / ts_, "%");
}

PolicyExecCtx::PolicyExecCtx(PolicyExecOpts& opts)
    : opts_(opts),
      use_cost_cache_(opts_.cost_policy ==
                      CostEstimationPolicy::kCachedExtrapolatedCost),
      fd_summ_(opts.env, GetLogPath(opts_.output_dir, opts_.policy_name, "summ")),
      fd_det_(opts.env, GetLogPath(opts_.output_dir, opts_.policy_name, "det")),
      fd_ranksum_(opts.env, GetLogPath(opts_.output_dir, opts_.policy_name, "ranksum")),
      ts_(0),
      ts_lb_invoked_(0),
      ts_lb_succeeded_(0),
      exec_time_us_(0) {
  Bootstrap();
}

void PolicyExecCtx::LogHeader(fort::char_table& table) {
  table << fort::header << "Name"
        << "LB Policy"
        << "Cost Policy"
        << "Trigger Policy"
        << "Timesteps";

  PolicyStats::LogHeader(table);

  table << "ExecTime" << fort::endr;
}

void PolicyExecCtx::LogSummary(fort::char_table& table) {
  table << opts_.policy_name << PolicyUtils::PolicyToString(opts_.lb_policy)
        << PolicyUtils::PolicyToString(opts_.cost_policy)
        << PolicyUtils::PolicyToString(opts_.trigger_policy)
        << std::to_string(ts_lb_succeeded_) + "/" +
               std::to_string(ts_lb_invoked_);

  stats_.LogSummary(table);

  table << PolicyStats::FormatProp(exec_time_us_ / 1e6, "s") << fort::endr;
}

void PolicyExecCtx::Bootstrap() {
  assert(opts_.nblocks_init % opts_.nranks == 0);
  int nblocks_per_rank = opts_.nblocks_init / opts_.nranks;

  lb_state_.ranklist.clear();

  for (int i = 0; i < opts_.nranks; ++i) {
    for (int j = 0; j < nblocks_per_rank; ++j) {
      lb_state_.ranklist.push_back(i);
    }
  }

  lb_state_.costlist_prev = std::vector<double>(opts_.nblocks_init, 1.0);

  assert(lb_state_.ranklist.size() == opts_.nblocks_init);
  logf(LOG_DBG2, "[PolicyExecCtx] Bootstrapping. Num Blocks: %d, Ranklist: %zu",
       opts_.nblocks_init, lb_state_.ranklist.size());
}

int PolicyExecCtx::ExecuteTimestep(std::vector<double> const& costlist_oracle,
                                   std::vector<int> const& ranklist_actual,
                                   std::vector<int>& refs,
                                   std::vector<int>& derefs) {
  int rv = 0;

  assert(lb_state_.costlist_prev.size() == lb_state_.ranklist.size());

  bool trigger_lb = ComputeLBTrigger(opts_.trigger_policy, lb_state_);
  if (trigger_lb) {
    std::vector<double> costlist_lb;
    ComputeCosts(ts_, costlist_oracle, costlist_lb);
    rv = TriggerLB(costlist_lb);
    if (rv) {
      logf(LOG_WARN, "[PolicyExecCtx] TriggerLB failed!");
      ts_++;
      return rv;
    }
  }

  // Timestep is always evaluated using the oracle cost
  if (opts_.lb_policy == LoadBalancePolicy::kPolicyActual) {
    assert(ranklist_actual.size() == costlist_oracle.size());
    logf(LOG_DBUG, "Logging with ranklist_actual (%zu)",
         ranklist_actual.size());
    stats_.LogTimestep(this, opts_.nranks, costlist_oracle, ranklist_actual);
  } else {
    assert(lb_state_.ranklist.size() == costlist_oracle.size());
    logf(LOG_DBUG, "Logging with lb.ranklist (%zu)", lb_state_.ranklist.size());
    stats_.LogTimestep(this, opts_.nranks, costlist_oracle, lb_state_.ranklist);
  }

  lb_state_.costlist_prev = costlist_oracle;
  lb_state_.refs = refs;
  lb_state_.derefs = derefs;

  ts_++;
  return rv;
}

int PolicyExecCtx::TriggerLB(const std::vector<double>& costlist) {
  int rv;
  std::vector<int> ranklist_lb;

  ts_lb_invoked_++;

  uint64_t lb_beg = pdlfs::Env::NowMicros();
  rv = LoadBalancePolicies::AssignBlocksInternal(opts_.lb_policy, costlist,
                                                 ranklist_lb, opts_.nranks);
  uint64_t lb_end = pdlfs::Env::NowMicros();

  if (rv) return rv;

  ts_lb_succeeded_++;
  exec_time_us_ += (lb_end - lb_beg);

  lb_state_.ranklist = ranklist_lb;

  assert(lb_state_.ranklist.size() == costlist.size());

  return rv;
}

std::string PolicyExecCtx::GetLogPath(const char* output_dir,
                                      const char* policy_name,
                                      const char* suffix) {
  std::regex rm_unsafe("[/-]");
  std::string result = std::regex_replace(policy_name, rm_unsafe, "_");
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  result = std::string(output_dir) + "/" + result + "." + suffix + ".csv";
  logf(LOG_DBUG, "LoadBalancePolicy Name: %s, Log Fname: %s", policy_name,
       result.c_str());
  return result;
}
};  // namespace amr
