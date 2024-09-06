#include "detailed_logger.h"

#include "logging.h"

const char* kOutlierEvents[] = {"Step 7: Pack and send buffers",
                                "Step 7a: MPI_Isend"};

const uint64_t kOutlierThresholds[] = {15000, 3000};

namespace amr {
TimestepwiseLogger::TimestepwiseLogger(pdlfs::WritableFile* fout, int rank)
    : fout_(fout),
      rank_(rank),
      metric_ids_(0),
      coalesce_(true),
      log_outliers_to_stderr_(false) {
  lines_.reserve(kFlushLimit);
  SetupOutliers();
};

void TimestepwiseLogger::SetupOutliers() {
  int nevents = sizeof(kOutlierEvents) / sizeof(kOutlierEvents[0]);
  int nthresh = sizeof(kOutlierThresholds) / sizeof(kOutlierThresholds[0]);

  if (nevents == 0) {
    logv(__LOG_ARGS__, LOG_INFO, "outlier logging disabled.");
    log_outliers_to_stderr_ = false;
    return;
  }

  if (nevents != nthresh) {
    logv(__LOG_ARGS__, LOG_ERRO,
         "outlier_events / outlier_thresholds mismatch: %d vs %d", nevents,
         nthresh);
    log_outliers_to_stderr_ = false;
    return;
  }

  log_outliers_to_stderr_ = true;

  for (int mid = 0; mid < nevents; mid++) {
    auto metric_id = GetMetricId(kOutlierEvents[mid]);
    outlier_thresholds_[metric_id] = kOutlierThresholds[mid];

    logvat0(__LOG_ARGS__, LOG_INFO,
            "mid: %d, outlier event: %s, threshold: %lu", metric_id,
            kOutlierEvents[mid], kOutlierThresholds[mid]);
  }
}

void TimestepwiseLogger::LogKey(const char* key, uint64_t val) {
  if (fout_ == nullptr) return;

  int metric_id = GetMetricId(key);
  bool coalesce = CoalesceKey(key);

  if (CoalesceKey(key)) {
    lines_.back().second += val;
  } else {
    lines_.push_back(Line(metric_id, val));
  }

  bool isFlushKey = false;
  if (strncmp(key, "MakeOutputs", 11) == 0) {
    isFlushKey = true;
  }

  // flush if necessary
  if ((lines_.size() >= kFlushLimit) and isFlushKey) {
    logvat0(__LOG_ARGS__, LOG_INFO, "Flushing %d data lines", lines_.size());
    HandleFlushing(fout_);
  }

  // outlier_handling
  if (!log_outliers_to_stderr_) return;

  auto it = outlier_thresholds_.find(metric_id);
  if (it == outlier_thresholds_.end()) return;

  if (val > it->second) {
    logv(__LOG_ARGS__, LOG_WARN, "[rank %d] outlier detected: %s %lu", rank_,
         key, val);
  }
}

}  // namespace amr
