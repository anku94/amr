//
// Created by Ankush J on 4/11/22.
//

#include "common.h"
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

namespace Globals {
int my_rank, nranks;
}

int logf(int lvl, const char *fmt, ...) {
  const char *prefix;
  va_list ap;
  switch (lvl) {
    case 5:
      prefix = " [Debug] ";
      break;
    case 4:
      prefix = " [Debug] ";
      break;
    case 3:
      prefix = "!!! ERROR !!! ";
      break;
    case 2:
      prefix = "-WARNING- ";
      break;
    case 1:
      prefix = "-INFO- ";
      break;
    default:
      prefix = "";
      break;
  }
  fprintf(stderr, "%s", prefix);

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\n");
  return 0;
}

int loge(const char *op, const char *path) {
  return logf(LOG_ERRO, "!%s(%s): %s", strerror(errno), op, path);
}

void msg_abort(int err, const char *msg, const char *func, const char *file,
               int line) {
  fputs("*** ABORT *** ", stderr);
  fprintf(stderr, "@@ %s:%d @@ %s] ", file, line, func);
  fputs(msg, stderr);
  if (err != 0) fprintf(stderr, ": %s (errno=%d)", strerror(err), err);
  fputc('\n', stderr);
  abort();
}