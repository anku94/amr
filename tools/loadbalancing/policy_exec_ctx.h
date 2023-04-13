//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include "common.h"
#include "lb_policies.h"
#include "policy.h"
#include "policy_stats.h"

#include <pdlfs-common/env.h>
#include <regex>

namespace amr {

class PolicyExecutionContext {
 public:
  PolicyExecutionContext(const char* policy_name, Policy policy,
                         pdlfs::Env* env)
      : policy_name_(policy_name),
        policy_(policy),
        env_(env),
        ts_(0),
        exec_time_us_(0),
        fd_(nullptr) {
    EnsureOutputFile();
  }

  ~PolicyExecutionContext() {
    if (fd_) {
      pdlfs::Status s;
      SAFE_IO(fd_->Close(), "Close failed");
    }
  }

  /*
   * @param cost_alloc Cost-vector for assignment by policy
   * @param cost_actual Cost-vector for load balance estimation
   */
  int ExecuteTimestep(int nranks, std::vector<double> const& cost_alloc,
                      std::vector<double> const& cost_actual) {
    int rv;
    int nblocks = cost_alloc.size();
    assert(nblocks == cost_actual.size());

    std::vector<int> rank_list(nblocks, -1);

    uint64_t ts_assign_beg = pdlfs::Env::NowMicros();
    rv = LoadBalancePolicies::AssignBlocksInternal(policy_, cost_alloc,
                                                   rank_list, nranks);
    uint64_t ts_assign_end = pdlfs::Env::NowMicros();
    if (rv) return rv;

    stats_.LogTimestep(nranks, fd_, cost_actual, rank_list);
    exec_time_us_ += (ts_assign_end - ts_assign_beg);
    ts_++;
    return rv;
  }

  void LogSummary() {
    logf(LOG_INFO, "Policy: %s (%d timesteps simulated)", policy_name_, ts_);
    logf(LOG_INFO, "-----------------------------------");
    stats_.LogSummary();
    logf(LOG_INFO, "\n\tExec Time: \t%.2f s\n", exec_time_us_ / 1e6);
  }

 private:
  void WriteHeader() {
    const char* header = "ts,avg_us,max_us\n";
    pdlfs::Status s;
    SAFE_IO(fd_->Append(header), "Write failed");
  }

  void WriteData(int ts, double avg, double max) {
    char buf[1024];
    int buf_len = snprintf(buf, 1024, " %d,%.0lf,%.0lf\n", ts, avg, max);
    pdlfs::Status s;
    SAFE_IO(fd_->Append(pdlfs::Slice(buf, buf_len)), "Write failed");
  }

  void EnsureOutputFile() {
    std::string fname = policy_name_;
    if (env_->FileExists(fname.c_str())) {
      logf(LOG_WARN, "Overwriting file: %s", fname.c_str());
      env_->DeleteFile(fname.c_str());
    }

    if (fd_ != nullptr) {
      logf(LOG_WARN, "File already exists!");
      return;
    }

    pdlfs::Status s = env_->NewWritableFile(fname.c_str(), &fd_);
    if (!s.ok()) {
      ABORT("Unable to open SequentialFile!");
    }

    WriteHeader();
  }

  std::string GetLogPath() const {
    std::regex rm_unsafe("/");
    std::string result = std::regex_replace(policy_name_, rm_unsafe, "_");
    logf(LOG_DBUG, "Policy Name: %s, Log Fname: %s", policy_name_,
         result.c_str());
    return result;
  }

  const char* const policy_name_;
  const Policy policy_;
  pdlfs::Env* const env_;

  int ts_;
  double exec_time_us_;
  pdlfs::WritableFile* fd_;

  PolicyStats stats_;

  friend class MiscTest;
};
}  // namespace amr