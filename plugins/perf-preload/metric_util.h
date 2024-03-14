#pragma once

#include "logging.h"
#include "metric.h"
#include "types.h"

#include <mpi.h>
#include <string>
#include <unordered_map>

namespace amr {
class CommonComputer {
 public:
  CommonComputer(StringVec& vec, int rank, int nranks)
      : rank_(rank), nranks_(nranks) {
    for (auto& s : vec) {
      str_map_[s] = 1;
    }

    if (rank_ == 0) {
      local_strings_ = vec;
    }
  }

  StringVec Compute() {
    int num_items = GetNumItems();
    if (num_items < 0) {
      return StringVec();
    }

    if (rank_ == 0) {
      Verbose(__LOG_ARGS__, 1, "Number of items: %d", num_items);
    }

    StringVec common_strings;
    for (int i = 0; i < num_items; ++i) {
      std::string key_global;
      if (rank_ == 0) {
        key_global = RunRound(local_strings_[i].c_str());
      } else {
        // Compiler doesn't like it if we pass a nullptr here
        key_global = RunRound("");
      }

      if (key_global.size() > 0) {
        common_strings.push_back(key_global);
      }
    }

    return common_strings;
  }

 private:
  const int rank_, nranks_;
  std::unordered_map<std::string, int> str_map_;
  StringVec local_strings_;

  int GetNumItems() {
    int num_items = 0;
    if (rank_ == 0) {
      num_items = local_strings_.size();
    }

    int rv = PMPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "MPI_Bcast failed");
      return -1;
    }

    return num_items;
  }

  std::string RunRound(const char* key) {
    const size_t bufsz = 256;
    char buf[bufsz];
    memset(buf, 0, bufsz);

    if (key == nullptr and rank_ == 0) {
      Error(__LOG_ARGS__, "Key is null.");
      return "";
    }

    if (rank_ == 0) {
      Verbose(__LOG_ARGS__, 1, "Checking for key: %s", key);

      size_t key_len = strlen(key);
      if (key_len >= bufsz) {
        Error(__LOG_ARGS__, "Key too long: %s. Will be truncated.", key);
      }

      strncpy(buf, key, bufsz);
      buf[bufsz - 1] = 0;
    }

    int rv = PMPI_Bcast(buf, bufsz, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "MPI_Bcast failed");
      return "";
    }

    int exists_locally = (str_map_.find(buf) != str_map_.end()) ? 1 : 0;
    int exists_globally = 0;

    rv = PMPI_Allreduce(&exists_locally, &exists_globally, 1, MPI_INT, MPI_MIN,
                        MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      Error(__LOG_ARGS__, "MPI_Reduce failed");
      return "";
    }

    if (exists_globally) {
      if (rank_ == 0) {
        Verbose(__LOG_ARGS__, 1, "Key %s exists globally: %d", buf,
                exists_globally);
      }
      return std::string(buf);
    }

    Verbose(__LOG_ARGS__, 1, "Key %s does not exist globally.", buf);
    return "";
  }
};

class MetricCollectionUtils {
 public:
  template <typename T>
  static std::vector<T> GetVecByKey(StringVec& metrics, MetricMap& times,
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

#define SAFE_MPI_REDUCE(...)                     \
  {                                              \
    int rv = PMPI_Reduce(__VA_ARGS__);           \
    if (rv != MPI_SUCCESS) {                     \
      Error(__LOG_ARGS__, "PMPI_Reduce failed"); \
    }                                            \
  }

  static std::vector<MetricStats> CollectMetrics(StringVec& metrics,
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
};
}  // namespace amr
