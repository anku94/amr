#include "common.h"
#include "logging.h"

#include <pdlfs-common/env.h>
#include <pdlfs-common/slice.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace amr {
class TimestepwiseLogger {
 public:
  TimestepwiseLogger(pdlfs::WritableFile* fout, int rank);

  void LogKey(const char* key, uint64_t val);

  ~TimestepwiseLogger() {
    if (fout_ == nullptr) return;

    HandleFinalFlushing(fout_);
    fout_->Close();
  }

 private:
  pdlfs::WritableFile* fout_;
  const int rank_;

  int metric_ids_;
  std::unordered_map<std::string, int> metrics_;

  typedef std::pair<int, uint64_t> Line;
  std::vector<Line> lines_;

  // if true, coalesce multiple consecutive writes to a key
  // mostly used for polling functions to reduce output size
  const bool coalesce_;

  // initialized to false, but automatically set to true if some
  // events are configured. events are currently hardcoded in an
  // array in the .cc file
  bool log_outliers_to_stderr_;
  std::unordered_map<int, uint64_t> outlier_thresholds_;

  static const int kFlushLimit = 65536;

  void SetupOutliers();

  int GetMetricId(const char* key) {
    auto it = metrics_.find(key);
    if (it == metrics_.end()) {
      metrics_[key] = metric_ids_++;
      return metric_ids_ - 1;
    }

    return it->second;
  }

  // all int-space ops on metric
  void LogKeyInner(int metric_id, uint64_t val, bool coalesce);

  void HandleFlushing(pdlfs::WritableFile* f) {
    FlushDataToFile(f);
    lines_.clear();
  }

  void HandleFinalFlushing(pdlfs::WritableFile* f) {
    if (!lines_.empty()) {
      FlushDataToFile(f);
      lines_.clear();
    }

    FlushMetricsToFile(f);
  }

  void FlushDataToFile(pdlfs::WritableFile* f) {
    logv(__LOG_ARGS__, LOG_DBG2, "Flushing %d data lines", lines_.size());

    for (auto& l : lines_) {
      char buf[256];
      int bufsz = snprintf(buf, sizeof(buf), "%d %lu\n", l.first, l.second);
      f->Append(pdlfs::Slice(buf, bufsz));
    }
  }

  void FlushMetricsToFile(pdlfs::WritableFile* f) {
    logv(__LOG_ARGS__, LOG_DBG2, "Flushing %d metric names", metrics_.size());

    for (auto& m : metrics_) {
      char buf[4096];
      int bufsz =
          snprintf(buf, sizeof(buf), "%d %s\n", m.second, m.first.c_str());
      f->Append(pdlfs::Slice(buf, bufsz));
    }
  }

  inline bool CoalesceKey(const char* key) {
    if (!coalesce_) return false;
    if (lines_.empty() or lines_.back().first != metrics_[key]) return false;
    if (strncmp(key, "MPI_All", 7) == 0) return false;
    if (strncmp(key, "MPI_Bar", 7) == 0) return false;

    return true;
  }
};
}  // namespace amr
