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

void decrement_log_level_once();

enum class Status { OK, MPIError, Error, InvalidPtr };

enum class MeshGenMethod {
  Ring,
  AllToAll,
  Dynamic,
  FromSingleTSTrace,
  FromMultiTSTrace
};

std::string MeshGenMethodToStr(MeshGenMethod t);

struct CommNeighbor {
  int block_id;
  int peer_rank;
  int msg_sz;
};

struct DriverOpts {
  MeshGenMethod meshgen_method;
  size_t blocks_per_rank;
  size_t size_per_msg;
  int comm_rounds; // number of rounds to repeat each topo for
  int comm_nts;    // only used for trace mode
  const char *trace_root;
  const char *bench_log;
  const char *job_dir;

  DriverOpts()
      : meshgen_method(MeshGenMethod::Ring), blocks_per_rank(SIZE_MAX),
        size_per_msg(SIZE_MAX), comm_rounds(-1), comm_nts(-1), trace_root("") {}

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
    NA_IF(meshgen_method == MeshGenMethod::FromSingleTSTrace);
    NA_IF(meshgen_method == MeshGenMethod::FromMultiTSTrace);
    INVALID_IF(blocks_per_rank == SIZE_MAX);
    INVALID_IF(size_per_msg == SIZE_MAX);
    INVALID_IF(comm_rounds == -1);
    INVALID_IF(bench_log == nullptr);
    INVALID_IF(job_dir == nullptr);
    IS_VALID();
  }

  bool IsValidFromTrace() {
    NA_IF(meshgen_method != MeshGenMethod::FromSingleTSTrace and
          meshgen_method != MeshGenMethod::FromMultiTSTrace);
    INVALID_IF(trace_root == nullptr);
    INVALID_IF(bench_log == nullptr);
    INVALID_IF(job_dir == nullptr);
    IS_VALID();
  }

#undef NA_IF
#undef INVALID_IF
#undef IS_VALID
};

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
