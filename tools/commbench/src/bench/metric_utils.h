#pragma once

#include <sys/stat.h>

#include <sstream>
#include <string>
#include <vector>

namespace topo {
using ExtraMetric = std::pair<std::string, std::string>;
using ExtraMetricVec = std::vector<ExtraMetric>;

class MetricUtils {
 public:
  struct MetricData {
    std::vector<std::string> header;
    std::vector<std::string> fmtdata_csv;
    std::vector<std::string> fmtdata_print;

    // AddMetric: add a metric to the metric data
    void AddMetric(std::string key, std::string val_csv,
                   std::string val_print = "") {
      header.push_back(key);
      fmtdata_csv.push_back(val_csv);
      if (!val_print.empty()) {
        fmtdata_print.push_back(val_print);
      } else {
        fmtdata_print.push_back(val_csv);
      }
    }
  };

  // JoinVec: join a vector of strings with a delimiter
  static std::string JoinVec(const std::vector<std::string> &vec,
                             const std::string &delim) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i != vec.size() - 1) {
        oss << delim;
      }
    }
    return oss.str();
  };

  // WriteMetricData: write the metric data to a file
  // - If the file does not exist, create it and write the header
  // - Append the data to the file
  static void WriteMetricData(const char *file, MetricData const &md) {
    bool file_exists = FileExists(file);

    FILE *f = fopen(file, "a+");
    if (f == nullptr) return;

    if (!file_exists) {
      std::string header_str = JoinVec(md.header, ",");
      fprintf(f, "%s\n", header_str.c_str());
    }

    std::string data_str = JoinVec(md.fmtdata_csv, ",");
    fprintf(f, "%s\n", data_str.c_str());

    fclose(f);
  }

  // FileExists: check if a file exists
  static bool FileExists(const char *file) {
    struct stat statbuf;
    return stat(file, &statbuf) == 0;
  }
};
}  // namespace topo
