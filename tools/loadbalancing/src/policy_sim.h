#pragma once

#include "lb_policies.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <string>
#include <vector>

namespace amr {
class PolicyExecutionContext;

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

struct ExecCtxWrapper {
  const char* const ctx_name;
  Policy policy;
  bool unit_cost;  // If true, use unit cost for all blocks
  PolicyExecutionContext ctx;

  ExecCtxWrapper(const char* ctx_name, Policy policy, bool unit_cost,
                 const PolicySimOptions& sim_opts)
      : ctx_name(ctx_name),
        policy(policy),
        unit_cost(unit_cost),
        ctx(sim_opts.output_dir.c_str(), ctx_name, policy, sim_opts.env,
            sim_opts.nranks) {}
};

class PolicySim {
 public:
  explicit PolicySim(const PolicySimOptions& options)
      : options_(options), env_(options.env), nts_(0), bad_ts_(0) {}

  void Run() {
    logf(LOG_INFO, "Using prof dir: %s", options_.prof_dir.c_str());
    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());

    EnsureOutputDir();

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

    policies_.emplace_back(policy_name.c_str(), Policy::kPolicyILP, false,
                           options_);

    int shard_beg = shard_idx * num_ts / num_shards;
    int shard_end = std::min((shard_idx + 1) * num_ts / num_shards, num_ts);

    logf(LOG_INFO, "[Policy %s] Simulating trace from %d to %d",
         policy_name.c_str(), shard_beg, shard_end);

    ts_beg = shard_beg;
    ts_end = shard_end;
  }

 private:
  void EnsureOutputDir();

  void SetupAll();

  std::vector<std::string> LocateTraceFiles(
      const std::string& search_dir) const;

  void SimulateTrace(int ts_beg_ = 0, int ts_end_ = INT_MAX);

  void LogSummary();

  void LogSummary(fort::char_table& table);

  int InvokePolicies(std::vector<int>& cost_actual);

  static std::vector<std::string> FilterByRegex(
      std::vector<std::string>& strings, std::string regex_pattern) {
    std::vector<std::string> matches;
    const std::regex regex_obj(regex_pattern);

    for (auto& s : strings) {
      std::smatch match_obj;
      if (std::regex_match(s, match_obj, regex_obj)) {
        matches.push_back(s);
      }
    }
    return matches;
  }

  const PolicySimOptions options_;
  pdlfs::Env* const env_;
  std::vector<ExecCtxWrapper> policies_;

  int nts_;
  int bad_ts_;
};
}  // namespace amr
