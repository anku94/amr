#pragma once

#include "policy.h"
#include "prof_reader.h"
#include "trace_utils.h"

namespace amr {
class ProfSetReader {
 public:
  explicit ProfSetReader(const std::vector<std::string>& fpaths,
                         ProfTimeCombinePolicy combine_policy)
      : nblocks_prev_(0) {
    logf(LOG_INFO, "[ProfSetReader] Combine Policy: %s",
         Utils::GetProfTimeCombinePolicyStr(combine_policy).c_str());

    for (auto& fpath : fpaths) {
      all_readers_.emplace_back(fpath.c_str(), combine_policy);
    }
  }

  int ReadTimestep(std::vector<int>& times) {
    std::fill(times.begin(), times.end(), 0);

    if (times.size() < nblocks_prev_) {
      times.resize(nblocks_prev_, 0);
    }

    int nblocks = 0;

    for (auto& reader : all_readers_) {
      std::fill(times.begin(), times.end(), 0);
      int rnblocks = reader.ReadNextTimestep(times);
      nblocks = std::max(nblocks, rnblocks);
    }

    logf(LOG_DBUG, "Blocks read: %d", nblocks);

    if (nblocks > 0) {
      nblocks_prev_ = nblocks;
    }

    // LogTime upsizes the vector if necessary
    // This is to downsize the vector if necessary
    times.resize(nblocks);

    return nblocks;
  }

  int ReadTimestep(int timestep, std::vector<int>& times) {
    std::fill(times.begin(), times.end(), 0);

    int nblocks = 0;

    for (auto& reader : all_readers_) {
      int nlines_read = 0;
      int rnblocks = reader.ReadTimestep(timestep, times, nlines_read);
      nblocks = std::max(nblocks, rnblocks);
    }

    logf(LOG_DBUG, "Blocks read: %d", nblocks);

    if (nblocks > 0) {
      nblocks_prev_ = nblocks;
    }

    // LogTime upsizes the vector if necessary
    // This is to downsize the vector if necessary
    times.resize(nblocks);

    return nblocks;
  }

 private:
  std::vector<ProfileReader> all_readers_;
  int nblocks_prev_;
};
}  // namespace amr
