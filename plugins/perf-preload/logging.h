/*
 * Copyright (c) 2019 Carnegie Mellon University,
 * Copyright (c) 2019 Triad National Security, LLC, as operator of
 *     Los Alamos National Laboratory.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */
#pragma once

#include <pdlfs-common/env.h>

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
  if (rank_ == 0) {                                                 \
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
