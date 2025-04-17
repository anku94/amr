#pragma once

#include "common.h"

namespace Globals {
extern int my_rank, nranks;
extern DriverOpts driver_opts;
}; // namespace Globals

// #define EXPAND_ARGS(...) __VA_ARGS__
//
// #define logvat0_expand(info_log, file, line, level, fmt, ...)                  \
//   if (Globals::my_rank == 0) {                                                 \
//     logv(info_log, file, line, level, fmt, ##__VA_ARGS__);                     \
//   } else {                                                                     \
//     logv(info_log, file, line, level + 1, fmt, ##__VA_ARGS__);                 \
//   }
//
// #define logvat0(...) EXPAND_ARGS(logvat0_expand(__VA_ARGS__))
