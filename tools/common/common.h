//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <iomanip>
#include <pdlfs-common/env.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

#define LOG_ERRO 1
#define LOG_WARN 2
#define LOG_INFO 3
#define LOG_DBUG 4
#define LOG_DBG2 5
#define LOG_DBG3 6

#define DEF_LOGGER ::pdlfs::Logger::Default()
#define __LOG_ARGS__ DEF_LOGGER, __FILE__, __LINE__

int logv(pdlfs::Logger *info_log, const char *file, int line, int level,
         const char *fmt, ...);
int loge(const char *op, const char *path);

#define EXPAND_ARGS(...) __VA_ARGS__

#define logvat0_expand(info_log, file, line, level, fmt, ...)                  \
  if (Globals::my_rank == 0) {                                                 \
    logv(info_log, file, line, level, fmt, ##__VA_ARGS__);                     \
  } else {                                                                     \
    logv(info_log, file, line, level + 1, fmt, ##__VA_ARGS__);                 \
  }

#define logvat0(...) EXPAND_ARGS(logvat0_expand(__VA_ARGS__))

/*
 * logging facilities and helpers
 */
#define ABORT_FILENAME                                                         \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define ABORT(msg) msg_abort(errno, msg, __func__, ABORT_FILENAME, __LINE__)

/* abort with an error message */
void msg_abort(int err, const char *msg, const char *func, const char *file,
               int line);

enum class Status { OK, MPIError, Error, InvalidPtr };

enum class NeighborTopology {
  Ring,
  AllToAll,
  Dynamic,
  FromSingleTSTrace,
  FromMultiTSTrace
};

std::string TopologyToStr(NeighborTopology t);

struct CommNeighbor {
  int block_id;
  int peer_rank;
  int msg_sz;
};

struct DriverOpts {
  NeighborTopology topology;
  int topology_nbrcnt; // XXX: Is this used anywhere?
  size_t blocks_per_rank;
  size_t size_per_msg;
  int comm_rounds; // number of rounds to repeat each topo for
  int comm_nts;    // only used for trace mode
  const char *trace_root;
  const char *bench_log;
  const char *job_dir;

  DriverOpts()
      : topology(NeighborTopology::Ring), topology_nbrcnt(-1),
        blocks_per_rank(SIZE_MAX), size_per_msg(SIZE_MAX), comm_rounds(-1),
        comm_nts(-1), trace_root("") {}

#define NA_IF(x)                                                               \
  if (x)                                                                       \
    return true;
#define INVALID_IF(x)                                                          \
  if (x)                                                                       \
    return false;
#define IS_VALID() return true;

  /* all constituent Invalid checks return True if N/A.
   * Therefore all need to pass */
  bool IsValid() { return IsValidGeneric() && IsValidFromTrace(); }

private:
  bool IsValidGeneric() {
    NA_IF(topology == NeighborTopology::FromSingleTSTrace);
    NA_IF(topology == NeighborTopology::FromMultiTSTrace);
    INVALID_IF(blocks_per_rank == SIZE_MAX);
    INVALID_IF(size_per_msg == SIZE_MAX);
    INVALID_IF(comm_rounds == -1);
    INVALID_IF(bench_log == nullptr);
    INVALID_IF(job_dir == nullptr);
    IS_VALID();
  }

  bool IsValidFromTrace() {
    NA_IF(topology != NeighborTopology::FromSingleTSTrace
        and topology != NeighborTopology::FromMultiTSTrace);
    INVALID_IF(trace_root == nullptr);
    INVALID_IF(comm_nts == -1);
    INVALID_IF(bench_log == nullptr);
    INVALID_IF(job_dir == nullptr);
    IS_VALID();
  }

#undef NA_IF
#undef INVALID_IF
#undef IS_VALID
};

namespace Globals {
extern int my_rank, nranks;
extern DriverOpts driver_opts;
}; // namespace Globals

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
std::string SerializeVector(std::vector<T> const &v, int trunc_count = -1) {
  std::stringstream ss;

  ss << std::setprecision(3) << std::fixed;

  if (v.empty()) {
    ss << "<empty>";
  } else {
    ss << "(" << v.size() << " items): ";
  }

  int idx = 0;
  ss << std::endl;
  for (auto n : v) {
    ss << n << " ";
    idx++;
    if (trunc_count == idx) {
      ss << "... <truncated>";
      break;
    }

    if (idx % 10 == 0) {
      ss << std::endl;
    }
  }

  return ss.str();
}
