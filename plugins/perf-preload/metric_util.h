#pragma once

#include "logging.h"
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

    logv(__LOG_ARGS__, LOG_DBUG, "[Rank %d] Number of items: %d", rank_,
         num_items);

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
      logv(__LOG_ARGS__, LOG_ERRO, "MPI_Bcast failed");
      return -1;
    }

    return num_items;
  }

  std::string RunRound(const char* key) {
    const size_t bufsz = 256;
    char buf[bufsz];
    memset(buf, 0, bufsz);

    if (key == nullptr and rank_ == 0) {
      logv(__LOG_ARGS__, LOG_ERRO, "Key is null.");
      return "";
    }

    if (rank_ == 0) {
      logv(__LOG_ARGS__, LOG_DBUG, "Checking for key: %s", key);

      size_t key_len = strlen(key);
      if (key_len >= bufsz) {
        logv(__LOG_ARGS__, LOG_ERRO, "Key too long: %s. Will be truncated.",
             key);
      }

      strncpy(buf, key, bufsz);
      buf[bufsz - 1] = 0;
    }

    int rv = PMPI_Bcast(buf, bufsz, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      logv(__LOG_ARGS__, LOG_ERRO, "MPI_Bcast failed");
      return "";
    }

    int exists_locally = (str_map_.find(buf) != str_map_.end()) ? 1 : 0;
    int exists_globally = 0;

    rv = PMPI_Allreduce(&exists_locally, &exists_globally, 1, MPI_INT, MPI_MIN,
                        MPI_COMM_WORLD);
    if (rv != MPI_SUCCESS) {
      logv(__LOG_ARGS__, LOG_ERRO, "MPI_Reduce failed");
      return "";
    }

    if (exists_globally) {
      if (rank_ == 0) {
        logv(__LOG_ARGS__, LOG_DBG2, "Key %s exists globally: %d", buf,
             exists_globally);
      }
      return std::string(buf);
    }

    logv(__LOG_ARGS__, LOG_DBG2, "Key %s does not exist globally.", buf);
    return "";
  }
};
}  // namespace amr
