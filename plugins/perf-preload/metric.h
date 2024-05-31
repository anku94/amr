#pragma once

#include "logging.h"
#include "types.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <mpi.h>
#include <sstream>

#define SAFE_MPI_REDUCE(...)           \
  {                                    \
    int rv = PMPI_Reduce(__VA_ARGS__); \
    if (rv != MPI_SUCCESS) {           \
      ABORT("PMPI_Reduce failed");     \
    }                                  \
  }

namespace amr {
class MetricCollectionUtils;
struct MetricStats {
  std::string name;
  uint64_t invoke_count;
  double min;
  double max;
  double avg;
  double std;
};

/* Metric: Maintains and aggregates summary statistics for a single metric
 *
 * Stats tracked: min/max/avg/std
 * Internal representation uses double-precision floats to track sums of
 * squares etc. without overflow risk.
 *
 * Methods
 * - LogInstance: log a single instance of the metric
 * - Collect: collect summary statistics using MPI_Reduce
 */
class Metric {
 public:
  Metric(const char* name, int rank)
      : name_(name),
        rank_(rank),
        invoke_count_(0),
        sum_val_(0.0),
        max_val_(DBL_MIN),
        min_val_(DBL_MAX),
        sum_sq_val_(0.0) {}

  void LogInstance(double val) {
    invoke_count_++;
    sum_val_ += val;
    max_val_ = std::max(max_val_, val);
    min_val_ = std::min(min_val_, val);
    sum_sq_val_ += val * val;
  }

 public:
  std::string GetMetricRankwise(int nranks) const {
    std::ostringstream oss;

    oss << name_ << "\n";

    auto metric = GetMetricRankwiseInner(nranks);
    if (rank_ != 0) {
      return "";
    }

    for (int i = 0; i < nranks; ++i) {
      if (i == 0) {
        oss << metric[i];
      } else {
        oss << "," << metric[i];
      }
    }

    oss << "\n";
    return oss.str();
  }

  static std::vector<MetricStats> CollectMetrics(StringVec const& metrics,
                                                 MetricMap& times) {

    size_t nmetrics = metrics.size();

    auto get_invoke_cnt = [](Metric& m) { return m.invoke_count_; };
    auto invoke_vec = GetVecByKey<uint64_t>(metrics, times, get_invoke_cnt);
    std::vector<uint64_t> global_invoke_cnt(nmetrics, 0);

    SAFE_MPI_REDUCE(invoke_vec.data(), global_invoke_cnt.data(), nmetrics,
                    MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    auto get_sum = [](Metric& m) { return m.sum_val_; };
    auto sum_vec = GetVecByKey<double>(metrics, times, get_sum);
    std::vector<double> global_sum(nmetrics, 0);
    SAFE_MPI_REDUCE(sum_vec.data(), global_sum.data(), nmetrics, MPI_DOUBLE,
                    MPI_SUM, 0, MPI_COMM_WORLD);

    auto get_max = [](Metric& m) { return m.max_val_; };
    auto max_vec = GetVecByKey<double>(metrics, times, get_max);
    std::vector<double> global_max(nmetrics, 0);
    SAFE_MPI_REDUCE(max_vec.data(), global_max.data(), nmetrics, MPI_DOUBLE,
                    MPI_MAX, 0, MPI_COMM_WORLD);

    auto get_min = [](Metric& m) { return m.min_val_; };
    auto min_vec = GetVecByKey<double>(metrics, times, get_min);
    std::vector<double> global_min(nmetrics, 0);
    SAFE_MPI_REDUCE(min_vec.data(), global_min.data(), nmetrics, MPI_DOUBLE,
                    MPI_MIN, 0, MPI_COMM_WORLD);

    auto get_sum_sq = [](Metric& m) { return m.sum_sq_val_; };
    auto sum_sq_vec = GetVecByKey<double>(metrics, times, get_sum_sq);
    std::vector<double> global_sum_sq(nmetrics, 0);
    SAFE_MPI_REDUCE(sum_sq_vec.data(), global_sum_sq.data(), nmetrics,
                    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<MetricStats> stats;

    for (size_t i = 0; i < nmetrics; ++i) {
      MetricStats s;
      s.name = metrics[i];

      s.invoke_count = global_invoke_cnt[i];
      s.avg = global_sum[i] / global_invoke_cnt[i];
      s.max = global_max[i];
      s.min = global_min[i];

      auto var = (global_sum_sq[i] / global_invoke_cnt[i]) - (s.avg * s.avg);
      s.std = sqrt(var);
      stats.push_back(s);
    }

    return stats;
  }

 private:
  const std::string name_;
  const int rank_;

  uint64_t invoke_count_;

  double sum_val_;
  double max_val_;
  double min_val_;
  double sum_sq_val_;

  template <typename T>
  static std::vector<T> GetVecByKey(StringVec const& metrics, MetricMap& times,
                                    std::function<T(Metric&)> f) {
    std::vector<T> vec;
    for (auto& m : metrics) {
      auto it = times.find(m);
      if (it != times.end()) {
        vec.push_back(f(it->second));
      }
    }
    return vec;
  }

  std::vector<double> GetMetricRankwiseInner(int nranks) const {
    std::vector<double> metric(nranks, 0);
    metric[rank_] = sum_val_;

    if (rank_ == 0) {
      SAFE_MPI_REDUCE(MPI_IN_PLACE, metric.data(), nranks, MPI_DOUBLE, MPI_SUM,
                      0, MPI_COMM_WORLD);
    } else {
      SAFE_MPI_REDUCE(metric.data(), nullptr, nranks, MPI_DOUBLE, MPI_SUM, 0,
                      MPI_COMM_WORLD);
    }
    return metric;
  }
};
}  // namespace amr
