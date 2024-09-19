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

/*
 * Copyright (c) 2011 The LevelDB Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found at https://github.com/google/leveldb.
 */
#include "logging.h"
#include <glog/logging.h>

int logv(pdlfs::Logger *info_log, const char *file, int line, int level,
         const char *fmt, ...) {
  static bool first_time = true;
  if (first_time) {
    google::InitGoogleLogging("amrmon");
    first_time = false;
  }

  va_list ap;
  va_start(ap, fmt);
  if (info_log != NULL) {
    int glvl = 0, gvlvl = 0; // glog level and glog verbose lvl
    if (level == LOG_ERRO) {
      glvl = 2;
    } else if (level == LOG_WARN) {
      glvl = 1;
    } else if (level >= LOG_INFO) {
      glvl = 0;
      gvlvl = level - LOG_INFO;
    }

    info_log->Logv(file, line, glvl, gvlvl, fmt, ap);
  }
  va_end(ap);

  return 0;
}

int loge(const char *op, const char *path) {
  return logv(__LOG_ARGS__, LOG_ERRO, "!%s(%s): %s", strerror(errno), op, path);
}

void msg_abort(int err, const char *msg, const char *func, const char *file,
               int line) {
  fputs("*** ABORT *** ", stderr);
  fprintf(stderr, "@@ %s:%d @@ %s] ", file, line, func);
  fputs(msg, stderr);
  if (err != 0)
    fprintf(stderr, ": %s (errno=%d)", strerror(err), err);
  fputc('\n', stderr);
  abort();
}
