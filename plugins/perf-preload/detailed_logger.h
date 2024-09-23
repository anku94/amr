#include "common.h"
#include "logging.h"

#include <pdlfs-common/env.h>
#include <pdlfs-common/slice.h>
#include <string>
#include <unordered_map>
#include <vector>

// static const uint64_t kInitTimestamp = pdlfs::CurrentMicros();

namespace amr {
struct MetricWithTimestamp {
  uint64_t ts_micros;
  int metric_id;
  bool is_open;
  uint64_t duration;

  static uint64_t CurrentNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
  }

  MetricWithTimestamp(int metric_id, bool is_open, uint64_t duration)
      : ts_micros(CurrentNs()),
        metric_id(metric_id),
        is_open(is_open),
        duration(duration) {}

  void Serialize(char* buf, int bufsz) {
    int n = snprintf(buf, bufsz, "%lu %d %d %lu\n", ts_micros, metric_id,
                     is_open ? 1 : 0, duration);
    if (n >= bufsz) {
      logv(__LOG_ARGS__, LOG_ERRO, "Buffer overflow in MetricWithTimestamp");
    }
  }

  void UpdateDuration(uint64_t d) {
    duration += d;
    ts_micros = CurrentNs();
  }
};

class TimestepwiseLogger {
 public:
  TimestepwiseLogger(pdlfs::WritableFile* fout, int rank);

  ~TimestepwiseLogger() {
    if (fout_ == nullptr) return;

    HandleFinalFlushing(fout_);
    fout_->Close();
  }

  void FinalFlush() {
    if (fout_ == nullptr) return;

    HandleFinalFlushing(fout_);
    fout_->Close();
    fout_ = nullptr;
  }

  void LogBegin(const char* key);

  void LogEnd(const char* key, uint64_t duration);

 private:
  pdlfs::WritableFile* fout_;
  const int rank_;

  int metric_ids_;
  std::unordered_map<std::string, int> metrics_;

  typedef std::pair<int, uint64_t> Line;

  std::vector<MetricWithTimestamp> metric_lines_;

  // if true, coalesce multiple consecutive writes to a key
  // mostly used for polling functions to reduce output size
  const bool coalesce_;

  static const int kFlushLimit = 65536;

  int flush_key_count_ = 0;

  int GetMetricId(const char* key) {
    auto it = metrics_.find(key);
    if (it == metrics_.end()) {
      metrics_[key] = metric_ids_++;
      return metric_ids_ - 1;
    }

    return it->second;
  }

  void HandleFlushing(pdlfs::WritableFile* f) {
    FlushTimestampDataToFile(f);
    metric_lines_.clear();
  }

  void HandleFinalFlushing(pdlfs::WritableFile* f) {
    if (!metric_lines_.empty()) {
      FlushTimestampDataToFile(f);
      metric_lines_.clear();
    }

    FlushMetricsToFile(f);
  }

  void FlushTimestampDataToFile(pdlfs::WritableFile* f) {
    logv(__LOG_ARGS__, LOG_DBG2, "Flushing %d metric lines",
         metric_lines_.size());

    for (auto& l : metric_lines_) {
      char buf[256];
      l.Serialize(buf, sizeof(buf));
      f->Append(pdlfs::Slice(buf, strlen(buf)));
      f->Flush();
    }
  }

  void FlushMetricsToFile(pdlfs::WritableFile* f) {
    logv(__LOG_ARGS__, LOG_DBG2, "Flushing %d metric names", metrics_.size());

    for (auto& m : metrics_) {
      char buf[4096];
      int bufsz =
          snprintf(buf, sizeof(buf), "%d %s\n", m.second, m.first.c_str());
      f->Append(pdlfs::Slice(buf, bufsz));
      f->Flush();
    }
  }

  inline bool CoalesceStackKey(const char* key) {
    if (!coalesce_) return false;
    if (metric_lines_.size() < 2) return false;
    if (strncmp(key, "MPI_All", 7) == 0) return false;
    if (strncmp(key, "MPI_Bar", 7) == 0) return false;
    if (strncmp(key, "Mesh::", 6) == 0) return false;

    int metric_id = GetMetricId(key);
    // return true iff last two entries are the same metric, one is open, the
    // other is close
    auto metric_last = metric_lines_.back();
    auto metric_last2 = metric_lines_[metric_lines_.size() - 2];

    if (metric_last.metric_id == metric_id and
        metric_last2.metric_id == metric_id and metric_last2.is_open and
        !metric_last.is_open) {
      return true;
    }

    return false;
  }
};
}  // namespace amr
