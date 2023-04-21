//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <random>
#include <sstream>
#include <stdio.h>
#include <string.h>

// TODO: log_levels should probably be ordered the other way
#define LOG_LEVEL 2

#define LOG_ERRO 5
#define LOG_WARN 4
#define LOG_INFO 3
#define LOG_DBUG 2
#define LOG_DBG2 1

int logf(int lvl, const char* fmt, ...);
int loge(const char* op, const char* path);

/*
 * logging facilities and helpers
 */
#define ABORT_FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define ABORT(msg) msg_abort(errno, msg, __func__, ABORT_FILENAME, __LINE__)

/* abort with an error message */
void msg_abort(int err, const char* msg, const char* func, const char* file,
               int line);

enum class Status { OK, MPIError, Error, InvalidPtr };

enum class NeighborTopology { Ring, AllToAll, Dynamic, FromTrace };

std::string TopologyToStr(NeighborTopology t);

struct DriverOpts {
  NeighborTopology topology;
  int topology_nbrcnt;  // XXX: Is this used anywhere?
  size_t blocks_per_rank;
  size_t size_per_msg;
  int comm_rounds;
  const char* trace_root;

  DriverOpts()
      : topology(NeighborTopology::Ring),
        topology_nbrcnt(-1),
        blocks_per_rank(SIZE_MAX),
        size_per_msg(SIZE_MAX),
        comm_rounds(-1),
        trace_root("") {}

#define NA_IF(x) \
  if (x) return true;
#define INVALID_IF(x) \
  if (x) return false;
#define IS_VALID() return true;

  /* all constituent Invalid checks return True if N/A.
   * Therefore all need to pass */
  bool IsValid() { return IsValidGeneric() && IsValidFromTrace(); }

 private:
  bool IsValidGeneric() {
    NA_IF(topology == NeighborTopology::FromTrace);
    INVALID_IF(blocks_per_rank == SIZE_MAX);
    INVALID_IF(size_per_msg == SIZE_MAX);
    INVALID_IF(comm_rounds == -1);
    IS_VALID();
  }

  bool IsValidFromTrace() {
    NA_IF(topology != NeighborTopology::FromTrace);
    INVALID_IF(trace_root == nullptr);
    IS_VALID();
  }

#undef NA_IF
#undef INVALID_IF
#undef IS_VALID
};

namespace Globals {
extern int my_rank, nranks;
extern DriverOpts driver_opts;
};  // namespace Globals

class NormalGenerator {
 public:
  NormalGenerator(double mean, double std)
      : mean_(mean), std_(std), gen_(rd_()), d_(mean_, std_) {}

  double Generate() { return d_(gen_); }

  float GenFloat() { return d_(gen_); }

  int GenInt() { return std::round(d_(gen_)); }

 private:
  const double mean_;
  const double std_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<> d_;
};

template <typename T>
std::string SerializeVector(std::vector<T> const& v, int trunc_count = -1) {
  std::stringstream ss;
  ss << "(" << v.size() << " items): ";

  int idx = 0;
  for (auto n : v) {
    ss << n << " ";
    idx++;
    if (trunc_count == idx) {
      ss << "... <truncated>";
      break;
    }
  }

  return ss.str();
}
