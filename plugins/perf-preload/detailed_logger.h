#include <pdlfs-common/env.h>
#include <pdlfs-common/slice.h>
#include <string>
#include <unordered_map>
#include <vector>

class TimestepwiseLogger {
 public:
  TimestepwiseLogger(pdlfs::WritableFile* fout)
      : fout_(fout), metric_ids_(0), coalesce_(true) {
    lines_.reserve(kFlushLimit);
  }

  void LogKey(const char* key, uint64_t val) {
    if (fout_ == nullptr) return;

    auto it = metrics_.find(key);
    if (it == metrics_.end()) {
      metrics_[key] = metric_ids_++;
    }

    if (CoalesceKey(key)) {
      lines_.back().second += val;
    } else {
      lines_.push_back(Line(metrics_[key], val));
    }

    if (lines_.size() >= kFlushLimit) {
      HandleFlushing(fout_);
    }
  }

  ~TimestepwiseLogger() {
    if (fout_ == nullptr) return;

    HandleFinalFlushing(fout_);
    fout_->Close();
  }

 private:
  pdlfs::WritableFile* fout_;
  typedef std::pair<int, uint64_t> Line;
  int metric_ids_;
  std::unordered_map<std::string, int> metrics_;
  std::vector<Line> lines_;
  // if true, coalesce multiple consecutive writes to a key
  // mostly used for polling functions to reduce output size
  const bool coalesce_;

  static const int kFlushLimit = 65536;

  void HandleFlushing(pdlfs::WritableFile* f) {
    if (lines_.size() >= kFlushLimit) {
      FlushDataToFile(f);
      lines_.clear();
    }
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
