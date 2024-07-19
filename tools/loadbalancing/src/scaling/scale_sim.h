//
// Created by Ankush J on 7/13/23.
//

#pragma once

#include <mpi.h>

#include <utility>

#include "constants.h"
#include "distrib/distributions.h"
#include "policy_utils.h"
#include "run_utils.h"
#include "scale_stats.h"
#include "tabular_data.h"
#include "trace_utils.h"

namespace amr {
struct ScaleSimOpts {
  std::string output_dir;
  pdlfs::Env* env;
  int nblocks_beg;
  int nblocks_end;
  int nranks;
};

struct RunProfile {
  int nranks;
  int nblocks;
};

class ScaleSim {
 public:
  explicit ScaleSim(ScaleSimOpts opts) : options_(std::move(opts)) {}

  // fake_rank - only used to control logging - don't use for MPI purposes
  void RunSuite(std::vector<std::string>& suite, RunProfile const& rp,
                std::vector<double>& costs, MPI_Comm comm, int fake_rank) {
    int nruns = suite.size();

    std::vector<int> ranks(rp.nranks, 0);

    for (auto& policy : suite) {
      logv(__LOG_ARGS__, LOG_INFO, "[RUN] %s", policy.c_str());
    }

    logv(__LOG_ARGS__, LOG_INFO, "Using output dir: %s",
         options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);

    for (auto& policy : suite) {
      // begin timing
      uint64_t _ts_beg = options_.env->NowMicros();

      RunType r = {rp.nranks, rp.nblocks, policy};
      for (int iter = 0; iter < Constants::kScaleSimIters; iter++) {
        int rv = 0;

        if (comm == MPI_COMM_NULL) {
          rv = r.AssignBlocks(costs, ranks, rp.nranks);
        } else {
          rv = r.AssignBlocksParallel(costs, ranks, rp.nranks, comm);
        }

        if (rv) {
          ABORT("Failed to assign blocks");
        }
      }

      // end timing
      uint64_t _ts_end = options_.env->NowMicros();

      RunType::VerifyAssignment(ranks, rp.nranks);

      // compute stats and log them
      std::vector<double> rank_times;
      double time_avg = 0, time_max = 0;

      PolicyUtils::ComputePolicyCosts(rp.nranks, costs, ranks, rank_times,
                                      time_avg, time_max);
      if (fake_rank == 0) {
        logv(__LOG_ARGS__, LOG_INFO,
             "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
             r.policy.c_str(), time_avg, time_max);
      }

      double iter_time = (_ts_end - _ts_beg) * 1.0 / Constants::kScaleSimIters;
      double loc_cost = PolicyUtils::ComputeLocCost(ranks) * 100;

      std::shared_ptr<TableRow> row = std::make_shared<ScaleSimRow>(
          r.policy, rp.nblocks, rp.nranks, iter_time, time_avg, time_max,
          loc_cost);

      table_.addRow(row);
    }
  }

  void Run() {
    logv(__LOG_ARGS__, LOG_INFO, "Using output dir: %s",
         options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);

    std::vector<RunProfile> run_profiles;  // = {{2048, 8192}, {4096, 8192}};
    GenRunProfiles(run_profiles);
    std::vector<double> costs;

    std::vector<std::string> policy_suite = {
        "baseline", "cdp", "hybrid25",     "hybrid50",
        "hybrid75", "lpt", "hybrid75alt1", "hybrid75alt2"};

    policy_suite = {"baseline", "cdp", "cdpc512", "hybrid50", "lpt"};
    policy_suite = {"cdp", "cdpc512", "lpt"};

    int nruns = policy_suite.size();

    for (auto& r : run_profiles) {
      logv(__LOG_ARGS__, LOG_INFO,
           "[Running profile] nranks_: %d, nblocks: %d, iters: %d, nruns: %d",
           r.nranks, r.nblocks, Constants::kScaleSimIters, nruns);

      if (costs.size() != r.nblocks) {
        costs.resize(r.nblocks, 0);
        auto opts = DistributionUtils::GetConfigOpts();
        DistributionUtils::GenDistribution(opts, costs, r.nblocks);
      }

      RunSuite(policy_suite, r, costs, MPI_COMM_NULL, 0);
    }

    EmitTable(nruns);
  }

  void RunParallel(MPI_Comm comm) {
    int my_rank = -1;
    MPI_Comm_rank(comm, &my_rank);

    logvat0(my_rank, __LOG_ARGS__, LOG_INFO, "Using output dir: %s",
            options_.output_dir.c_str());

    if (my_rank == 0) {
      Utils::EnsureDir(options_.env, options_.output_dir);
    }

    std::vector<RunProfile> run_profiles;  // = {{2048, 8192}, {4096, 8192}};
    GenRunProfiles(run_profiles);
    std::vector<double> costs;

    std::vector<std::string> policy_suite = {
        "cdp",          "cdpc512", "cdpc512par1", "cdpc512par4", "cdpc512par8",
        "cdpc512par16", "lpt",     "hybrid25",    "hybrid50",    "hybrid75"};
    policy_suite = {"cdp", "cdpc512", "cdpc512par8" }; //, "hybrid25", "hybrid50", "hybrid75", "lpt"};
    policy_suite = { "cdp", "cdpc512", "cdpc512par8", "hybrid25" };
    // policy_suite = { "cdpc512par8" };

    int nruns = policy_suite.size();

    for (auto& r : run_profiles) {
      logv(__LOG_ARGS__, LOG_INFO,
           "[Running profile] nranks_: %d, nblocks: %d, iters: %d, nruns: %d",
           r.nranks, r.nblocks, Constants::kScaleSimIters, nruns);

      if (costs.size() != r.nblocks) {
        costs.resize(r.nblocks, 0);
        GenDistributionParallel(costs, r.nblocks, comm);
      }

      MPI_Barrier(comm);
      RunSuite(policy_suite, r, costs, comm, my_rank);
    }

    if (my_rank == 0) {
      EmitTable(nruns);
    }
  }

 private:
  void GenDistributionParallel(std::vector<double>& costs, int nblocks,
                               MPI_Comm comm) {
    int my_rank = -1;
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank == 0) {
      auto opts = DistributionUtils::GetConfigOpts();
      DistributionUtils::GenDistribution(opts, costs, nblocks);
    }

    int rv = MPI_Bcast(costs.data(), nblocks, MPI_DOUBLE, 0, comm);
    if (rv != MPI_SUCCESS) {
      logv(__LOG_ARGS__, LOG_ERRO, "MPI_Bcast failed");
      ABORT("MPI_Bcast failed");
    }
  }

  void EmitTable(int n) {
    std::string table_out = options_.output_dir + "/scalesim.log.csv";
    std::stringstream table_stream;
    table_.emitTable(table_stream, n);
    logv(__LOG_ARGS__, LOG_INFO, "Table: \n%s", table_stream.str().c_str());

    Utils::WriteToFile(options_.env, table_out, table_.toCSV());
  }

  void GenRunProfiles(std::vector<RunProfile>& v) {
    v.clear();

    int blcnt_beg = options_.nblocks_beg;
    int blcnt_end = options_.nblocks_end;
    int nranks = options_.nranks;

    for (int nblocks = blcnt_beg; nblocks <= blcnt_end; nblocks *= 2) {
      if (nranks != -1) {
        v.emplace_back(RunProfile{nranks, nblocks});
        continue;
      }

      int nranks_init = GetSmallestPowerBiggerThanN(2, nblocks / 5);
      for (int nranks = nranks_init; nranks <= nblocks; nranks *= 2) {
        v.emplace_back(RunProfile{nranks, nblocks});
      }
    }

    for (auto& rp : v) {
      logv(__LOG_ARGS__, LOG_INFO, "RunProfile: nranks: %d, nblocks: %d",
           rp.nranks, rp.nblocks);
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
