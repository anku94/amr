//
// Created by Ankush J on 4/11/22.
//

#include "common.h"

#include <errno.h>
#include <glog/logging.h>
#include <pdlfs-common/env.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

std::string MeshGenMethodToStr(MeshGenMethod t) {
  switch (t) {
  case MeshGenMethod::Ring:
    return "RING";
  case MeshGenMethod::AllToAll:
    return "ALL_TO_ALL";
  case MeshGenMethod::Dynamic:
    return "DYNAMIC";
  case MeshGenMethod::FromSingleTSTrace:
    return "FROM_SINGLE_TS_TRACE";
  case MeshGenMethod::FromMultiTSTrace:
    return "FROM_MULTI_TS_TRACE";
  }

  return "UNKNOWN";
}

int logv(pdlfs::Logger *info_log, const char *file, int line, int level,
         const char *fmt, ...) {
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

void decrement_log_level_once() {
  static bool _first = true;
  if (!_first)
    return;
  _first = false;

  if (FLAGS_v > 0) {
    FLAGS_v--;
  } else if (FLAGS_minloglevel < 3) {
    FLAGS_v = 0;
    FLAGS_minloglevel++;
  }
}
