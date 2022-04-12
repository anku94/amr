//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <stdio.h>

#define LOG_DBG2 5
#define LOG_DBUG 4
#define LOG_ERRO 3
#define LOG_WARN 2
#define LOG_INFO 1

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

enum class Status {
  OK,
  MPIError,
  Error,
  InvalidPtr
};

namespace Globals {
extern int my_rank, nranks;
};
enum class NeighborTopology { Ring,
                              AllToAll };

struct DriverOpts {
  NeighborTopology topology;
  size_t blocks_per_rank;
  size_t msgs_per_block;
  size_t size_per_msg;
};
