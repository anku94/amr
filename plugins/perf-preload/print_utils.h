#include "p2p.h"

#include <string>

namespace amr {
class MetricPrintUtils {
 public:
  static std::string GetHeader() {
    int bufsz = 256;
    char buf[bufsz];

    const char* header_fmt = "%68s: %12s %12s %12s %12s %12s\n";
    snprintf(buf, bufsz, header_fmt, "Metric", "Count", "Avg", "Std", "Min",
             "Max");

    std::string dashes(68 + 12 * 5 + 6, '-');
    return std::string(buf) + dashes + "\n";
  }

  static std::string GetMetricLine(const char* metric_name,
                                   uint64_t invoke_count, double avg,
                                   double std, double min, double max) {
    int bufsz = 256;
    char buf[bufsz];

    const char* metric_fmt = "%68s: %12d %12.0lf %12.0lf %12.0lf %12.0lf\n";
    snprintf(buf, bufsz, metric_fmt, metric_name, invoke_count, avg, std, min,
             max);

    return std::string(buf);
  }

  static std::string CountMatrixAnalysisToStr(const char* matrix_name,
                                              const MatrixAnalysis& ma) {
    std::string ret = "";

    int bufsz = 256;
    char buf[bufsz];

    const char* header_fmt = "%20s %12s %12s %12s\n";
    snprintf(buf, bufsz, header_fmt, "Matrix", "Local", "Global", "Pct");

    ret += std::string(buf);

    const char* matrix_fmt = "%20s %12s %12s %6.2lf%%\n";
    auto local_str = CountToStr(ma.sum_local);
    auto global_str = CountToStr(ma.sum_global);
    double pct_local = 100.0 * ma.sum_local / ma.sum_global;
    snprintf(buf, bufsz, matrix_fmt, matrix_name, local_str.c_str(),
             global_str.c_str(), pct_local);

    ret += std::string(buf);

    return ret;
  }

  static std::string SizeMatrixAnalysisToStr(const char* matrix_name,
                                             const MatrixAnalysis& ma) {
    std::string ret = "";

    int bufsz = 256;
    char buf[bufsz];

    const char* header_fmt = "%20s %12s %12s %12s\n";
    snprintf(buf, bufsz, header_fmt, "Matrix", "Local", "Global", "Pct");

    ret += std::string(buf);

    const char* matrix_fmt = "%20s %12s %12s %6.2lf%%\n";
    auto local_str = BytesToStr(ma.sum_local);
    auto global_str = BytesToStr(ma.sum_global);
    double pct_local = 100.0 * ma.sum_local / ma.sum_global;
    snprintf(buf, bufsz, matrix_fmt, matrix_name, local_str.c_str(),
             global_str.c_str(), pct_local);

    ret += std::string(buf);

    return ret;
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
    snprintf(buf, bufsz, "%.1lf %s", size_d, units.c_str());

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
    snprintf(buf, bufsz, "%.1lf %s", size_d, units.c_str());

    return std::string(buf);
  }
};
}  // namespace amr
