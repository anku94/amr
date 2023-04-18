//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include "common.h"
#include "lb_policies.h"
#include "lb_trigger.h"
#include "policy.h"
#include "policy_stats.h"

#include <pdlfs-common/env.h>
#include <regex>

namespace amr {

class PolicyExecutionContext {
 public:
  PolicyExecutionContext(const char* output_dir, const char* policy_name,
                         Policy policy, pdlfs::Env* env, int nranks)
      : output_dir_(output_dir),
        policy_name_(policy_name),
        policy_(policy),
        env_(env),
        nranks_(nranks),
        ts_invoked_(0),
        ts_triggered_(0),
        ts_succeeded_(0),
        exec_time_us_(0),
        fd_(nullptr),
        trigger_(nranks) {
    EnsureOutputFile();
  }

  // Safe move constructor for fd_
  PolicyExecutionContext(PolicyExecutionContext&& rhs) noexcept
      : output_dir_(rhs.output_dir_),
        policy_name_(rhs.policy_name_),
        policy_(rhs.policy_),
        env_(rhs.env_),
        nranks_(rhs.nranks_),
        ts_invoked_(rhs.ts_invoked_),
        ts_triggered_(rhs.ts_triggered_),
        ts_succeeded_(rhs.ts_succeeded_),
        exec_time_us_(rhs.exec_time_us_),
        fd_(rhs.fd_),
        trigger_(std::move(rhs.trigger_)) {
    if (this != &rhs) {
      rhs.fd_ = nullptr;
    }
  }

  PolicyExecutionContext(const PolicyExecutionContext& rhs) = delete;

  PolicyExecutionContext& operator=(PolicyExecutionContext&& rhs) = delete;

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
  int ExecuteTimestep(std::vector<double> const& cost_alloc,
                      std::vector<double> const& cost_actual) {
    ts_invoked_++;

    int rv = 0;
    int nblocks = cost_alloc.size();
    assert(nblocks == cost_actual.size());

    if (trigger_.Trigger(cost_alloc)) {
      ts_triggered_++;

      std::vector<int> rank_list(nblocks, -1);

      uint64_t ts_assign_beg = pdlfs::Env::NowMicros();
      rv = LoadBalancePolicies::AssignBlocksInternal(policy_, cost_alloc,
                                                     rank_list, nranks_);
      uint64_t ts_assign_end = pdlfs::Env::NowMicros();

      if (rv) return rv;
      ts_succeeded_++;
      
      exec_time_us_ += (ts_assign_end - ts_assign_beg);
      stats_.LogTimestep(nranks_, fd_, cost_actual, rank_list);
    } else {
      std::vector<int> rank_list = trigger_.GetLastAssignment();
      stats_.LogTimestep(nranks_, fd_, cost_actual, rank_list);
    }

    ts_succeeded_++;
    return rv;
  }

  void LogSummary() {
    logf(LOG_INFO, "Policy: %s (%d/%d timesteps simulated)", policy_name_,
         ts_succeeded_, ts_invoked_);
    logf(LOG_INFO, "-----------------------------------");
    stats_.LogSummary();
    logf(LOG_INFO, "\n\tExec Time: \t%.2f s\n", exec_time_us_ / 1e6);
  }

  void LogSummary(fort::char_table& table) {
    table << policy_name_
          << std::to_string(ts_succeeded_) + "/" + std::to_string(ts_invoked_);
    stats_.LogSummary(table);
    table << PolicyStats::FormatProp(exec_time_us_ / 1e6, "s") << fort::endr;
  }

  std::string Name() const { return policy_name_; }

 private:
  void EnsureOutputFile() {
    std::string fname = GetLogPath(output_dir_, policy_name_);
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
      ABORT("Unable to open WriteableFile!");
    }
  }

  static std::string GetLogPath(const char* output_dir,
                                const char* policy_name) {
    std::regex rm_unsafe("[/-]");
    std::string result = std::regex_replace(policy_name, rm_unsafe, "_");
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    result = std::string(output_dir) + "/" + result + ".csv";
    logf(LOG_DBUG, "Policy Name: %s, Log Fname: %s", policy_name,
         result.c_str());
    return result;
  }

  const char* const output_dir_;
  const char* const policy_name_;
  const Policy policy_;
  pdlfs::Env* const env_;
  const int nranks_;

  int ts_invoked_;
  int ts_triggered_;
  int ts_succeeded_;
  double exec_time_us_;
  pdlfs::WritableFile* fd_;

  PolicyStats stats_;

  LoadBalanceTrigger trigger_;

  friend class MiscTest;
};
}  // namespace amr