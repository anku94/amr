#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

class Histogram {
 public:
  static std::string Plot(const std::vector<double>& data) {
    if (data.empty()) return "";

    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    int min_bin = std::floor(min_val / 10) * 10;
    int max_bin = std::ceil(max_val / 10) * 10;
    int bin_count = (max_bin - min_bin) / 10 + 1;

    std::vector<int> bins(bin_count, 0);
    for (float v : data) ++bins[(int((v - min_bin) / 10))];

    int total_count = data.size();
    int max_freq = *std::max_element(bins.begin(), bins.end());
    std::ostringstream oss;

    for (int i = 0; i < bin_count; ++i) {
      int bin_val = min_bin + i * 10;
      int count = bins[i];
      double pct = (count * 100.0) / total_count;
      int stars = max_freq ? count * 10 / max_freq : 0;

      oss << "binval:" << std::setw(4) << std::setfill('0') << bin_val << " |";
      oss << std::string(stars, '*') << std::string(10 - stars, ' ') << "| ";
      oss << std::setw(6) << count << " / " << std::setw(6) << total_count
          << " (";
      oss << std::fixed << std::setprecision(2) << std::setw(5) << pct
          << "%)\n";
    }
    return oss.str();
  }
};
