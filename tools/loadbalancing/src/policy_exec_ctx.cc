//
// Created by Ankush J on 5/4/23.
//

#include "policy_exec_ctx.h"

#include "fort.hpp"
#include "policy.h"

namespace amr {
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
      fd_(nullptr),
      ts_(0),
      ts_lb_invoked_(0),
      ts_lb_succeeded_(0),
      exec_time_us_(0) {
  EnsureOutputFile();
  Bootstrap();
}

PolicyExecCtx::PolicyExecCtx(PolicyExecCtx&& rhs) noexcept
    : opts_(rhs.opts_),
      cost_cache_(rhs.cost_cache_),
      use_cost_cache_(rhs.use_cost_cache_),
      lb_state_(rhs.lb_state_),
      stats_(rhs.stats_),
      fd_(rhs.fd_),
      ts_(rhs.ts_),
      ts_lb_invoked_(rhs.ts_lb_invoked_),
      ts_lb_succeeded_(rhs.ts_lb_succeeded_),
      exec_time_us_(rhs.exec_time_us_) {
  if (this != &rhs) {
    rhs.fd_ = nullptr;
  }
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

int PolicyExecCtx::ExecuteTimestep(const std::vector<double>& costlist_oracle,
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

  assert(lb_state_.ranklist.size() == costlist_oracle.size());

  // Timestep is always evaluated using the oracle cost
  stats_.LogTimestep(opts_.nranks, fd_, costlist_oracle, lb_state_.ranklist);

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

void PolicyExecCtx::EnsureOutputFile() {
  std::string fname = GetLogPath(opts_.output_dir, opts_.policy_name);
  if (opts_.env->FileExists(fname.c_str())) {
    logf(LOG_WARN, "Overwriting file: %s", fname.c_str());
    opts_.env->DeleteFile(fname.c_str());
  }

  if (fd_ != nullptr) {
    logf(LOG_WARN, "File already exists!");
    return;
  }

  pdlfs::Status s = opts_.env->NewWritableFile(fname.c_str(), &fd_);
  if (!s.ok()) {
    ABORT("Unable to open WriteableFile!");
  }
}

std::string PolicyExecCtx::GetLogPath(const char* output_dir,
                                      const char* policy_name) {
  std::regex rm_unsafe("[/-]");
  std::string result = std::regex_replace(policy_name, rm_unsafe, "_");
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  result = std::string(output_dir) + "/" + result + ".csv";
  logf(LOG_DBUG, "LoadBalancePolicy Name: %s, Log Fname: %s", policy_name,
       result.c_str());
  return result;
}
};  // namespace amr
