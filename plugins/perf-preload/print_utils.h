#pragma once

#include "metric.h"
#include "p2p.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace amr {
class MetricPrintUtils {
 public:
  static std::string GetHeader(size_t max_len) {
    int bufsz = 256;
    char buf[bufsz];

    std::string header_fmt =
        "%" + std::to_string(max_len) + "s: %12s %12s %12s %12s %12s\n";
    snprintf(buf, bufsz, header_fmt.c_str(), "Metric", "Count", "Avg", "Std",
             "Min", "Max");

    std::string dashes(max_len + 12 * 5 + 6, '-');
    return std::string(buf) + dashes + "\n";
  }

  static std::string GetMetricLine(size_t max_len, const char* metric_name,
                                   uint64_t invoke_count, double avg,
                                   double std, double min, double max) {
    int bufsz = 256;
    char buf[bufsz];

    std::string metric_fmt = "%" + std::to_string(max_len) +
                             "s: %12d %12.0lf %12.0lf %12.0lf %12.0lf\n";
    snprintf(buf, bufsz, metric_fmt.c_str(), metric_name, invoke_count, avg,
             std, min, max);

    return std::string(buf);
  }

  static std::string CommMatrixAnalysisHeader() {
    int bufsz = 256;
    char buf[bufsz];

    const char* header_fmt = "%20s %12s %12s %12s\n";
    snprintf(buf, bufsz, header_fmt, "Matrix", "Local", "Global", "PctLocal");

    std::string dashes(20 + 12 * 3 + 3, '-');
    return std::string(buf) + dashes + "\n";
  }

  static std::string CountMatrixAnalysisToStr(const char* matrix_name,
                                              const MatrixAnalysis& ma) {
    int bufsz = 256;
    char buf[bufsz];

    const char* matrix_fmt = "%20s %12s %12s %6.2lf%%\n";
    auto local_str = CountToStr(ma.sum_local);
    auto global_str = CountToStr(ma.sum_global);
    double pct_local = 0;

    if (ma.sum_global > 0) {
      pct_local = 100.0 * ma.sum_local / ma.sum_global;
    }

    snprintf(buf, bufsz, matrix_fmt, matrix_name, local_str.c_str(),
             global_str.c_str(), pct_local);

    return std::string(buf);
  }

  static std::string SizeMatrixAnalysisToStr(const char* matrix_name,
                                             const MatrixAnalysis& ma) {
    int bufsz = 256;
    char buf[bufsz];

    const char* matrix_fmt = "%20s %12s %12s %6.2lf%%\n";
    auto local_str = BytesToStr(ma.sum_local);
    auto global_str = BytesToStr(ma.sum_global);
    double pct_local = 0;

    if (ma.sum_global > 0) {
      pct_local = 100.0 * ma.sum_local / ma.sum_global;
    }

    snprintf(buf, bufsz, matrix_fmt, matrix_name, local_str.c_str(),
             global_str.c_str(), pct_local);

    return std::string(buf);
  }

  static std::string BytesToStr(uint64_t size) {
    std::string units;
    double size_d = size;

    if (size_d < 1024) {
      units = "B";
    } else if (size_d < 1024 * 1024) {
      size_d /= 1024;
      units = "KiB";
    } else if (size_d < 1024 * 1024 * 1024) {
      size_d /= 1024 * 1024;
      units = "MiB";
    } else {
      size_d /= 1024 * 1024 * 1024;
      units = "GiB";
    }

    int bufsz = 256;
    char buf[bufsz];
    snprintf(buf, bufsz, "%.2lf %s", size_d, units.c_str());

    return std::string(buf);
  }

  static std::string CountToStr(uint64_t count) {
    std::string units;
    double size_d = count;

    if (size_d < 1000) {
      units = "";
    } else if (size_d < 1000 * 1000) {
      size_d /= 1000;
      units = "K";
    } else if (size_d < 1000 * 1000 * 1000) {
      size_d /= 1000 * 1000;
      units = "M";
    } else {
      size_d /= 1000 * 1000 * 1000;
      units = "B";
    }

    int bufsz = 256;
    char buf[bufsz];
    snprintf(buf, bufsz, "%.2lf %s", size_d, units.c_str());

    return std::string(buf);
  }

  static std::string SortAndSerialize(std::vector<MetricStats>& all_stats,
                                      int top_k) {
    // sort by largest metric first
    auto comparator = [](const MetricStats& a, const MetricStats& b) {
      return a.avg > b.avg;
    };

    std::sort(all_stats.begin(), all_stats.end(), comparator);
    std::string ret;

    if (top_k > 0) {
      all_stats.resize(std::min(top_k, (int)all_stats.size()));
    }

    size_t stat_max_strlen = 0;
    for (auto& stats : all_stats) {
      stat_max_strlen = std::max(stat_max_strlen, stats.name.size());
    }

    ret += GetHeader(stat_max_strlen);

    for (auto& stats : all_stats) {
      ret += GetMetricLine(stat_max_strlen + 1, stats.name.c_str(),
                           stats.invoke_count, stats.avg, stats.std, stats.min,
                           stats.max);
    }

    return ret;
  }

  static std::string MatrixToStr(const std::vector<uint64_t>& matrix,
                                 int nranks) {
    size_t max_len = 0;

    for (auto val : matrix) {
      max_len = std::max(max_len, std::to_string(val).size());
    }

    return MatrixToStrUtil(matrix, nranks, max_len);
  }

  static std::string MatrixToStrUtil(const std::vector<uint64_t>& matrix,
                                     int nranks, size_t max_len) {
    std::ostringstream oss;
    for (int i = 0; i < nranks; i++) {
      for (int j = 0; j < nranks; j++) {
        oss << std::setw(max_len) << matrix[i * nranks + j] << " ";
      }
      oss << "\n";
    }

    return oss.str();
  }
};
}  // namespace amr
