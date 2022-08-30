//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <random>
#include <stdio.h>
#include <string.h>

#define LOG_LEVEL 3

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
struct DriverOpts {
  NeighborTopology topology;
  int topology_nbrcnt;
  size_t blocks_per_rank;
  size_t size_per_msg;
  int comm_rounds;
  const char* trace_root;
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
