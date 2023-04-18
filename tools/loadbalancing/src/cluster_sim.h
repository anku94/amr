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
    }

    nts_ = 0;
    return 0;
  }

  void HandleTimestep(const std::vector<int>& block_times_orig) {
    logf(LOG_DBUG, "Block times: %zu\n", block_times_orig.size());

    std::vector<int> block_times_new(block_times_orig.size());
    double rel_error = 0;
    int k = block_times_orig.size();

    int k_beg = 1;
    int k_end = k;
    int cur_k = k_beg + (k_end - k_beg) / 2;

    while (k_beg < k_end) {
      Cluster(block_times_orig, block_times_new, cur_k, rel_error);
      if (rel_error > 0.02) {
        k_beg = cur_k + 1;
      } else {
        k_end = cur_k;
      }
      cur_k = k_beg + (k_end - k_beg) / 2;
    }

    logf(LOG_INFO, "k: %d, rel_error: %f\n", cur_k, rel_error);
    WriteData(k, cur_k, rel_error);
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
  void EnsureOutputFile() {
    Utils::EnsureDir(options_.env, options_.output_dir);
    if (fd_) return;

    std::string fpath = options_.output_dir + "/cluster_sim.csv";

    pdlfs::Status s = options_.env->NewWritableFile(fpath.c_str(), &fd_);
    if (!s.ok()) {
      logf(LOG_ERRO, "Unable to open file: %s", s.ToString().c_str());
      ABORT("Unable to open file");
    }

    WriteHeader();
  }

  void WriteHeader() {
    EnsureOutputFile();
    std::string header = "ts,n,k,rel_error\n";
    pdlfs::Status s = fd_->Append(header);
    if (!s.ok()) {
      logf(LOG_ERRO, "Unable to write to file: %s", s.ToString().c_str());
      ABORT("Unable to write to file");
    }
  }

  void WriteData(int n, int k, double rel_error) {
    EnsureOutputFile();
    std::string data = std::to_string(nts_) + "," + std::to_string(n) + "," +
                       std::to_string(k) + "," + std::to_string(rel_error) +
                       "\n";
    pdlfs::Status s = fd_->Append(data);
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
