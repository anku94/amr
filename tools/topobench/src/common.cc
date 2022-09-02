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
DriverOpts driver_opts;
}

std::string TopologyToStr(NeighborTopology t) {
  switch (t) {
    case NeighborTopology::Ring:
      return "RING";
    case NeighborTopology::AllToAll:
      return "ALL_TO_ALL";
    case NeighborTopology::Dynamic:
      return "DYNAMIC";
    case NeighborTopology::FromTrace:
      return "FROM_TRACE";
  }

  return "UNKNOWN";
}

int logf(int lvl, const char* fmt, ...) {
  if (lvl < LOG_LEVEL) return 0;

  const char* prefix;
  va_list ap;
  switch (lvl) {
    case 5:
      prefix = "!!! ERROR !!! ";
      break;
    case 4:
      prefix = "-WARNING- ";
      break;
    case 3:
      prefix = "-INFO- ";
      break;
    case 2:
    case 1:
      prefix = " [Debug] ";
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

int loge(const char* op, const char* path) {
  return logf(LOG_ERRO, "!%s(%s): %s", strerror(errno), op, path);
}

void msg_abort(int err, const char* msg, const char* func, const char* file,
               int line) {
  fputs("*** ABORT *** ", stderr);
  fprintf(stderr, "@@ %s:%d @@ %s] ", file, line, func);
  fputs(msg, stderr);
  if (err != 0) fprintf(stderr, ": %s (errno=%d)", strerror(err), err);
  fputc('\n', stderr);
  abort();
}
