#pragma once

#include <unordered_map>
#include <vector>
#include "tools-common/logging.h"

namespace amr {
struct CachedAssignment {
  std::vector<int> ranklist;
  int reuse_count;
};

class AssignmentCache {
 public:
  AssignmentCache(int max_reuse) : max_reuse_(max_reuse) {}

  bool Get(int nblocks, std::vector<int>& ranklist) {
    MLOG(MLOG_DBG0, "Cache get (nblocks: %d)", nblocks);

    if (cache_.find(nblocks) == cache_.end()) {
      return false;
    }

    auto& entry = cache_[nblocks];
    if (entry.reuse_count >= max_reuse_) {
      MLOG(MLOG_DBG0, "Cache entry for %d blocks expired",
           nblocks);
      cache_.erase(nblocks);
      return false;
    }

    entry.reuse_count++;
    ranklist = entry.ranklist;

    MLOG(MLOG_DBG0, "Cache hit (nblocks: %d, new reusecnt: %d)",
         nblocks, entry.reuse_count);

    return true;
  }

  void Put(int nblocks, std::vector<int> const& ranklist) {
    MLOG(MLOG_DBG0, "Cache put (nblocks: %d)", nblocks);

    cache_[nblocks] = {ranklist, 0};
  }

 private:
  int max_reuse_;  // max cached allocation reuse count
  std::unordered_map<int, CachedAssignment> cache_;
};
}  // namespace amr
