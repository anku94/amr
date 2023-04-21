//
// Created by Ankush J on 4/17/23.
//

#pragma once

#include "cluster.h"
#include "prof_set_reader.h"
#include "utils.h"

#include <pdlfs-common/env.h>
#include <string>

namespace amr {
struct ClusterSimOptions {
  pdlfs::Env* env;
  std::string prof_dir;
  std::string output_dir;
};

class ClusterSim {
 public:
  explicit ClusterSim(ClusterSimOptions& options)
      : options_(options), nts_(0), fd_(nullptr) {}

  int Run() {
    ProfSetReader psr(Utils::LocateTraceFiles(options_.env, options_.prof_dir));

    std::vector<int> block_times;
    while (psr.ReadTimestep(block_times) > 0) {
      HandleTimestep(block_times);
      nts_++;
      break;
    }

    nts_ = 0;
    return 0;
  }

  void HandleTimestep(const std::vector<int>& block_times_orig) {
    logf(LOG_DBUG, "Block times: %zu\n", block_times_orig.size());

    std::vector<int> block_times_new(block_times_orig.size());
    double mean_rel_error, max_rel_error;
    int k = block_times_orig.size();

    int k_beg = 1;
    int k_end = k;
    int cur_k = k_beg + (k_end - k_beg) / 2;

    while (k_beg < k_end) {
      Cluster(block_times_orig, block_times_new, cur_k, mean_rel_error,
              max_rel_error);
//      BinSearchIterateBoundMean(mean_rel_error, max_rel_error, cur_k, k_beg,
//                                k_end);
      BinSearchIterateBoundMax(mean_rel_error, max_rel_error, cur_k, k_beg,
                               k_end);
    }

    Cluster(block_times_orig, block_times_new, cur_k, mean_rel_error,
            max_rel_error);

    logf(LOG_DBUG, "[ClusterSim] N: %d, K: %d, RelErr Mean: %.1f, Max: %.1f\n",
         k, cur_k, mean_rel_error, max_rel_error);
    logf(LOG_DBG2, "[ClusterSim] OrigTS: %s",
         SerializeVector(block_times_orig, /* trunc_count */ 50).c_str());
    logf(LOG_DBG2, "[ClusterSim] NewTS: %s",
         SerializeVector(block_times_new, /* trunc_count */ 50).c_str());

    WriteData(k, cur_k, mean_rel_error, max_rel_error);
  }

  ~ClusterSim() {
    if (fd_) {
      pdlfs::Status s = fd_->Close();
      if (!s.ok()) {
        logf(LOG_ERRO, "Unable to close file: %s", s.ToString().c_str());
      }
      fd_ = nullptr;
    }
  }

 private:
  static void BinSearchIterateBoundMax(double mean_rel_err, double max_rel_err,
                                       int& cur_k, int& k_beg, int& k_end) {
    if (max_rel_err > 0.02) {
      k_beg = cur_k + 1;
    } else {
      k_end = cur_k;
    }

    cur_k = k_beg + (k_end - k_beg) / 2;
  }

  static void BinSearchIterateBoundMean(double mean_rel_err, double max_rel_err,
                                        int& cur_k, int& k_beg, int& k_end) {
    if (mean_rel_err > 0.01) {
      k_beg = cur_k + 1;
    } else {
      k_end = cur_k;
    }

    cur_k = k_beg + (k_end - k_beg) / 2;
  }

  void EnsureOutputFile() {
    Utils::EnsureDir(options_.env, options_.output_dir);
    if (fd_) return;

    std::string fpath = options_.output_dir + "/cluster_sim_mean.csv";

    pdlfs::Status s = options_.env->NewWritableFile(fpath.c_str(), &fd_);
    if (!s.ok()) {
      logf(LOG_ERRO, "Unable to open file: %s", s.ToString().c_str());
      ABORT("Unable to open file");
    }

    WriteHeader();
  }

  void WriteHeader() {
    EnsureOutputFile();
    std::string header = "ts,n,k,mean_rel_error,max_rel_error\n";
    pdlfs::Status s = fd_->Append(header);
    if (!s.ok()) {
      logf(LOG_ERRO, "Unable to write to file: %s", s.ToString().c_str());
      ABORT("Unable to write to file");
    }
  }

  void WriteData(int n, int k, double mean_rel_error, double max_rel_error) {
    EnsureOutputFile();
    // std::string data = std::to_string(nts_) + "," + std::to_string(n) + "," +
    // std::to_string(k) + "," + std::to_string(rel_error) +
    // "\n";
    std::string data = std::to_string(nts_);
    data += "," + std::to_string(n);
    data += "," + std::to_string(k);
    data += "," + std::to_string(mean_rel_error);
    data += "," + std::to_string(max_rel_error);

    pdlfs::Status s = fd_->Append(data + "\n");
    if (!s.ok()) {
      logf(LOG_ERRO, "Unable to write to file: %s", s.ToString().c_str());
      ABORT("Unable to write to file");
    }
  }

  ClusterSimOptions options_;
  int nts_;
  pdlfs::WritableFile* fd_;
};
}  // namespace amr
