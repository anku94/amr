#include "detailed_logger.h"

#include "logging.h"

namespace amr {
TimestepwiseLogger::TimestepwiseLogger(pdlfs::WritableFile* fout, int rank)
    : fout_(fout),
      rank_(rank),
      metric_ids_(0),
      coalesce_(true) {
  metric_lines_.reserve(kFlushLimit);
};

void TimestepwiseLogger::LogBegin(const char* key) {
  if (fout_ == nullptr) return;

  int metric_id = GetMetricId(key);
  bool coalesce = CoalesceStackKey(key);

  if (!coalesce) {
    metric_lines_.push_back(MetricWithTimestamp(metric_id, true, 0));
  }
}

void TimestepwiseLogger::LogEnd(const char* key, uint64_t duration) {
  if (fout_ == nullptr) return;

  int metric_id = GetMetricId(key);
  bool coalesce = CoalesceStackKey(key);

  if (coalesce) {
    metric_lines_.back().UpdateDuration(duration);
  } else {
    metric_lines_.push_back(MetricWithTimestamp(metric_id, false, duration));
  }

  bool isFlushKey = false;
  if (strncmp(key, "MakeOutputs", 11) == 0) {
    isFlushKey = true;
  }

  if ((metric_lines_.size() >= kFlushLimit) and isFlushKey) {
    logvat0(__LOG_ARGS__, LOG_INFO, "Flushing %d metric lines",
            metric_lines_.size());
    HandleFlushing(fout_);
  }
}
}  // namespace amr
