#pragma once

#include "lb_policies.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"
#include "utils.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace amr {
class PolicyExecCtx;

struct PolicySimOptions {
  pdlfs::Env* env;
  std::string prof_dir;
  std::string output_dir;
  int nranks;
  bool sim_ilp;
  int num_ts;
  int ilp_shard_idx;
  int ilp_num_shards;

  PolicySimOptions()
      : env(nullptr),
        nranks(512),
        sim_ilp(false),
        num_ts(-1),
        ilp_shard_idx(-1),
        ilp_num_shards(-1) {}
};

class PolicySim {
 public:
  explicit PolicySim(PolicySimOptions options)
      : options_(std::move(options)), nts_(0), bad_ts_(0) {}

  void Run() {
    logf(LOG_INFO, "Using prof dir: %s", options_.prof_dir.c_str());
    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());

    Utils::EnsureDir(options_.env, options_.output_dir);

    int sim_ts_begin = 0;
    int sim_ts_end = INT_MAX;

    if (options_.sim_ilp) {
      logf(LOG_INFO, "[PolicySim] ILP only: %d/%d", options_.ilp_shard_idx,
           options_.ilp_num_shards);
      SetupILPShard(options_.num_ts, options_.ilp_shard_idx,
                    options_.ilp_num_shards, sim_ts_begin, sim_ts_end);
    } else {
      logf(LOG_INFO,
           "[PolicySim] Simulating all policies except ILP (all timesteps)");
      SetupAll();
    }

    SimulateTrace(sim_ts_begin, sim_ts_end);

    fort::char_table table;
    LogSummary(table);
  }

  void SetupILPShard(int num_ts, int shard_idx, int num_shards, int& ts_beg,
                     int& ts_end) {
    std::string policy_name = "ILP/Actual-Cost/Shard/" +
                              std::to_string(shard_idx) + "/" +
                              std::to_string(num_shards);

    PolicyExecOpts popts;
    popts.output_dir = options_.output_dir.c_str();
    popts.env = options_.env;
    popts.nranks = options_.nranks;
    popts.nblocks_init = 512;

    popts.SetPolicy(policy_name.c_str(), LoadBalancingPolicy::kPolicyILP,
                    CostEstimationPolicy::kOracleCost,
                    TriggerPolicy::kEveryTimestep);

    policies_.emplace_back(popts);

    int shard_beg = shard_idx * num_ts / num_shards;
    int shard_end = std::min((shard_idx + 1) * num_ts / num_shards, num_ts);

    logf(LOG_INFO, "[LoadBalancingPolicy %s] Simulating trace from %d to %d",
         policy_name.c_str(), shard_beg, shard_end);

    ts_beg = shard_beg;
    ts_end = shard_end;
  }

 private:
  static void SetupAll();

  void SimulateTrace(int ts_beg_ = 0, int ts_end_ = INT_MAX);

  void LogSummary(fort::char_table& table);

  int InvokePolicies(std::vector<int>& cost_oracle);

  const PolicySimOptions options_;
  std::vector<PolicyExecCtx> policies_;

  int nts_;
  int bad_ts_;
};
}  // namespace amr
