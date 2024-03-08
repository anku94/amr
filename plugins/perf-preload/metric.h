#pragma once

#include "logging.h"
#include "print_utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <mpi.h>

namespace amr {
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
  Metric()
      : invoke_count_(0),
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

  std::string Collect(const char* metric_name, int my_rank) {
    uint64_t global_invoke_count = 0;
    double global_sum_val = 0;
    double global_max_val = 0;
    double global_min_val = 0;
    double global_sum_sq_val = 0;

    int rv = PMPI_Reduce(&invoke_count_, &global_invoke_count, 1, MPI_INT,
                         MPI_SUM, 0, MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "Metric %s: PMPI_Reduce failed", metric_name);
      return "";
    }

    rv = PMPI_Reduce(&sum_val_, &global_sum_val, 1, MPI_DOUBLE, MPI_SUM, 0,
                     MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "Metric %s: PMPI_Reduce failed", metric_name);
      return "";
    }

    rv = PMPI_Reduce(&max_val_, &global_max_val, 1, MPI_DOUBLE, MPI_MAX, 0,
                     MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "Metric %s: PMPI_Reduce failed", metric_name);
      return "";
    }

    rv = PMPI_Reduce(&min_val_, &global_min_val, 1, MPI_DOUBLE, MPI_MIN, 0,
                     MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "Metric %s: PMPI_Reduce failed", metric_name);
      return "";
    }

    rv = PMPI_Reduce(&sum_sq_val_, &global_sum_sq_val, 1, MPI_DOUBLE, MPI_SUM,
                     0, MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "Metric %s: PMPI_Reduce failed", metric_name);
      return "";
    }

    if (my_rank != 0) {
      return "";
    }

    if (invoke_count_ == 0) {
      Error(__LOG_ARGS__, "Metric %s: no invocations", metric_name);
      return "";
    }

    double global_avg = global_sum_val / global_invoke_count;
    double global_var =
        (global_sum_sq_val / global_invoke_count) - (global_avg * global_avg);
    double global_std = sqrt(global_var);

    return MetricPrintUtils::GetMetricLine(metric_name, global_invoke_count,
                                           global_avg, global_std,
                                           global_min_val, global_max_val);
  }

 private:
  uint64_t invoke_count_;

  double sum_val_;
  double max_val_;
  double min_val_;
  double sum_sq_val_;
};
}  // namespace amr
