//
// Created by Ankush J on 7/13/23.
//

#pragma once

#include "constants.h"
#include "distrib/distributions.h"
#include "run_utils.h"
#include "scale_stats.h"
#include "tabular_data.h"
#include "trace_utils.h"

#include <utility>

namespace amr {
struct ScaleSimOpts {
  std::string output_dir;
  pdlfs::Env* env;
  int nblocks_beg;
  int nblocks_end;
};

struct RunProfile {
  int nranks;
  int nblocks;
};

class ScaleSim {
 public:
  explicit ScaleSim(ScaleSimOpts opts) : options_(std::move(opts)) {}

  void RunSuite(std::vector<RunType>& suite, RunProfile const& rp,
                std::vector<double>& costs) {
    int nruns = suite.size();

    std::vector<int> ranks(rp.nranks, 0);

    for (auto& r : suite) {
      logf(LOG_INFO, "[RUN] %s", r.ToString().c_str());
    }

    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);

    for (auto& r : suite) {
      uint64_t _ts_beg = options_.env->NowMicros();
      for (int iter = 0; iter < Constants::kScaleSimIters; iter++) {
        int rv = r.AssignBlocks(costs, ranks, r.nranks);
        if (rv) {
          ABORT("Failed to assign blocks");
        }
      }
      uint64_t _ts_end = options_.env->NowMicros();

      std::vector<double> rank_times;
      double time_avg = 0, time_max = 0;
      PolicyUtils::ComputePolicyCosts(r.nranks, costs, ranks, rank_times,
                                      time_avg, time_max);
      logf(LOG_INFO,
           "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
           r.policy_name.c_str(), time_avg, time_max);

      double iter_time = (_ts_end - _ts_beg) * 1.0 / Constants::kScaleSimIters;
      double loc_cost = PolicyUtils::ComputeLocCost(ranks) * 100;

      std::shared_ptr<TableRow> row = std::make_shared<ScaleSimRow>(
          r.policy_name, rp.nblocks, rp.nranks, iter_time, time_avg, time_max,
          loc_cost);

      table_.addRow(row);
    }
  }

  void Run() {
    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);

    std::vector<RunProfile> run_profiles;
    GenRunProfiles(run_profiles, options_.nblocks_beg, options_.nblocks_end);
    std::vector<double> costs;

    std::vector<RunType> suite = RunSuites::GetCppIterSuite(64, 150);
    int nruns = suite.size();

    for (auto& r : run_profiles) {
      logf(LOG_INFO,
           "[Running profile] nranks_: %d, nblocks: %d, iters: %d, nruns: %d",
           r.nranks, r.nblocks, Constants::kScaleSimIters, nruns);

      suite = RunSuites::GetCppIterSuite(r.nblocks, r.nranks);

      if (costs.size() != r.nblocks) {
        costs.resize(r.nblocks, 0);
        DistributionUtils::GenDistributionWithDefaults(costs, r.nblocks);
      }

      RunSuite(suite, r, costs);
    }

    EmitTable(nruns);
  }

 private:
  void EmitTable(int n) {
    std::string table_out = options_.output_dir + "/scalesim.log.csv";
    std::stringstream table_stream;
    table_.emitTable(table_stream, n);
    logf(LOG_INFO, "Table: \n%s", table_stream.str().c_str());

    Utils::WriteToFile(options_.env, table_out, table_.toCSV());
  }

  static void GenRunProfiles(std::vector<RunProfile>& v, int nb_beg,
                             int nb_end) {
    v.clear();

    for (int nblocks = nb_beg; nblocks <= nb_end; nblocks *= 2) {
      int nranks_init = GetSmallestPowerBiggerThanN(2, nblocks / 10);
      for (int nranks = nranks_init; nranks <= nblocks; nranks *= 2) {
        v.emplace_back(RunProfile{nranks, nblocks});
      }
    }
  }

  static int GetSmallestPowerBiggerThanN(int pow, int n) {
    int rv = 1;
    while (rv < n) {
      rv *= pow;
    }

    return rv;
  }

  ScaleSimOpts const options_;
  // std::vector<ScaleExecCtx> policies_;
  TabularData table_;
};
}  // namespace amr
