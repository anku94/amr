#pragma once

#include "policy.h"
#include "prof_base.h"
#include "prof_reader.h"
#include "trace_utils.h"

namespace amr {
class ProfSetReader {
 public:
  explicit ProfSetReader(const std::vector<std::string>& fpaths,
                         ProfTimeCombinePolicy combine_policy)
      : nblocks_prev_(0) {
    logv(__LOG_ARGS__, LOG_INFO, "[ProfSetReader] Combine Policy: %s",
         Utils::GetProfTimeCombinePolicyStr(combine_policy).c_str());

    for (auto& fpath : fpaths) {
      if (fpath.size() >= 4 and fpath.substr(fpath.size() - 4) == ".csv") {
        all_readers_.emplace_back(
            new CSVProfileReader(fpath.c_str(), combine_policy));
      } else {
        all_readers_.emplace_back(
            new BinProfileReader(fpath.c_str(), combine_policy));
      }
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
      int rnblocks = reader->ReadNextTimestep(times);
      nblocks = std::max(nblocks, rnblocks);
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Blocks read: %d", nblocks);

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
      int rnblocks = reader->ReadTimestep(timestep, times, nlines_read);
      nblocks = std::max(nblocks, rnblocks);
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Blocks read: %d", nblocks);

    if (nblocks > 0) {
      nblocks_prev_ = nblocks;
    }

    // LogTime upsizes the vector if necessary
    // This is to downsize the vector if necessary
    times.resize(nblocks);

    return nblocks;
  }

 private:
  std::vector<ProfileReader*> all_readers_;
  int nblocks_prev_;
};
}  // namespace amr
