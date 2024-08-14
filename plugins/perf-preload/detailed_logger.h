#include <pdlfs-common/env.h>
#include <pdlfs-common/slice.h>
#include <string>
#include <unordered_map>
#include <vector>

class TimestepwiseLogger {
 public:
  TimestepwiseLogger(pdlfs::WritableFile* fout) : fout_(fout), metric_ids_(0) {
    lines_.reserve(kFlushLimit);
  }

  void LogKey(const char* key, uint64_t val) {
    if (fout_ == nullptr) return;

    auto it = metrics_.find(key);
    if (it == metrics_.end()) {
      metrics_[key] = metric_ids_++;
    }

    lines_.push_back(Line(metrics_[key], val));

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
};