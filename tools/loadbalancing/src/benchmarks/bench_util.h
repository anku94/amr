#include "common.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <pdlfs-common/env.h>
#include <sstream>
#include <string>
#include <vector>

namespace amr {
class BenchmarkUtils {
 public:
  BenchmarkUtils(const std::string& output_dir, pdlfs::Env* env)
      : output_dir_(output_dir), env_(env) {}

  std::string GetPexFileName(const std::string& policy_name,
                             const std::string& distrib_name, int nranks,
                             int nblocks) {
    std::string policy_name_lower = policy_name;
    std::transform(policy_name_lower.begin(), policy_name_lower.end(),
                   policy_name_lower.begin(), ::tolower);

    std::string distrib_name_lower = distrib_name;
    std::transform(distrib_name_lower.begin(), distrib_name_lower.end(),
                   distrib_name_lower.begin(), ::tolower);

    std::string fname = policy_name_lower + "." + distrib_name_lower + "." +
                        std::to_string(nranks) + "." + std::to_string(nblocks) +
                        ".pex";

    return output_dir_ + "/" + fname;
  }

  void WritePexToFile(const std::string fpath,
                      const std::vector<double>& costlist,
                      const std::vector<int>& ranklist, const int nranks) {
    std::ostringstream ss;
    // write the entire costlist in one line, set precision to two decimals
    ss << std::fixed << std::setprecision(2);
    for (auto c : costlist) {
      ss << c << " ";
    }

    ss << "\n";
    // write the entire ranklist in one line
    for (auto r : ranklist) {
      ss << r << " ";
    }
    ss << "\n";
    // write nranks
    ss << nranks << "\n";

    WriteToFile(fpath, ss.str());
  }

  void WriteToFile(const std::string& fpath, const std::string& content) const {
    pdlfs::Status s = pdlfs::Status::OK();

    // ensure parent dir exists, create if it doesn't
    auto parent_dir = fpath.substr(0, fpath.find_last_of("/"));
    if (!env_->FileExists(parent_dir.c_str())) {
      s = env_->CreateDir(parent_dir.c_str());
      if (!s.ok()) {
        logv(__LOG_ARGS__, LOG_ERRO, "Error creating dir: %s", parent_dir.c_str());
        return;
      }
    }

    pdlfs::WritableFile* fh;

    s = env_->NewWritableFile(fpath.c_str(), &fh);
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_ERRO, "Error opening file: %s", fpath.c_str());
      return;
    }

    s = fh->Append(content);
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_ERRO, "Error opening file: %s", fpath.c_str());
    }

    s = fh->Close();
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_ERRO, "Error opening file: %s", fpath.c_str());
    }
  }

  template <typename T>
  static void LogVector(const std::string& name, const std::vector<T>& vec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "[Vector] " << name << ": \n";

    for (size_t idx = 0; idx < vec.size(); ++idx) {
      ss << std::setw(8) << vec[idx] << " ";
      if (idx % 10 == 9) {
        ss << "\n";
      }
    }

    logv(__LOG_ARGS__, LOG_DBG2, "%s", ss.str().c_str());
  }

  static void LogAllocation(int nblocks, int nranks,
                            const std::vector<double>& costlist,
                            const std::vector<int>& ranklist) {
    std::map<int, std::vector<int>> rank_to_block_costs;
    for (int i = 0; i < nblocks; i++) {
      rank_to_block_costs[ranklist[i]].push_back(costlist[i]);
    }

    for (auto& kv : rank_to_block_costs) {
      std::sort(kv.second.begin(), kv.second.end(), std::greater<int>());
    }

    std::ostringstream ss;
    ss << "[Allocation] \n";
    for (int ridx = 0; ridx < nranks; ridx++) {
      ss << "[Rank " << ridx << "] ";
      auto total = std::accumulate(rank_to_block_costs[ridx].begin(),
                                   rank_to_block_costs[ridx].end(), 0.0);
      ss << "(Total: " << std::setw(4) << total << ") ";
      for (auto c : rank_to_block_costs[ridx]) {
        ss << std::setw(4) << c << " ";
      }
      ss << "\n";
    }

    logv(__LOG_ARGS__, LOG_DBG2, "%s", ss.str().c_str());
  }

 private:
  const std::string output_dir_;
  pdlfs::Env* env_;
};
}  // namespace amr
