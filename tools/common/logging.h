#pragma once

#include <glog/logging.h> // IWYU pragma: export
#include <mpi.h>
#include <stdarg.h>

inline std::string fmtstr(int bufsz, const char *fmt, ...) {
  char buf[bufsz];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, bufsz, fmt, args);
  va_end(args);
  return std::string(buf);
}

#define MLOG_ERRO 0
#define MLOG_WARN 1
#define MLOG_INFO 2
#define MLOG_DBG0 3
#define MLOG_DBG1 4
#define MLOG_DBG2 5
#define MLOG_DBG3 6
#define MLOG_BUFSZ 1024

#define MLOG_INNER(lvl, fmt, ...)                                              \
  do {                                                                         \
    switch (lvl) {                                                             \
    case MLOG_ERRO:                                                            \
      LOG(ERROR) << fmtstr(MLOG_BUFSZ, "[ERRO] " fmt, ##__VA_ARGS__);          \
      break;                                                                   \
    case MLOG_WARN:                                                            \
      LOG(WARNING) << fmtstr(MLOG_BUFSZ, "[WARN] " fmt, ##__VA_ARGS__);        \
      break;                                                                   \
    case MLOG_INFO:                                                            \
      LOG(INFO) << fmtstr(MLOG_BUFSZ, "[INFO] " fmt, ##__VA_ARGS__);           \
      break;                                                                   \
    case MLOG_DBG0:                                                            \
      VLOG(0) << fmtstr(MLOG_BUFSZ, "[DBG0] " fmt, ##__VA_ARGS__);             \
      break;                                                                   \
    case MLOG_DBG1:                                                            \
      VLOG(1) << fmtstr(MLOG_BUFSZ, "[DBG1] " fmt, ##__VA_ARGS__);             \
      break;                                                                   \
    case MLOG_DBG2:                                                            \
      VLOG(2) << fmtstr(MLOG_BUFSZ, "[DBG2] " fmt, ##__VA_ARGS__);             \
      break;                                                                   \
    case MLOG_DBG3:                                                            \
    default:                                                                   \
      VLOG(3) << fmtstr(MLOG_BUFSZ, "[DBG3] " fmt, ##__VA_ARGS__);             \
      break;                                                                   \
    }                                                                          \
  } while (0);

#define MLOG(level, fmt, ...)                                                  \
  MLOG_INNER(level, "[%10.10s:%3.3d] " fmt,                                    \
             (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__), \
             __LINE__, ##__VA_ARGS__)

#define MLOGIF(cond, level, fmt, ...)                                          \
  if (cond) {                                                                  \
    MLOG(level, fmt, ##__VA_ARGS__)                                            \
  }

#define MLOGIFR0(level, fmt, ...)                                              \
  if (Globals::my_rank == 0) {                                                 \
    MLOG(level, fmt, ##__VA_ARGS__)                                            \
  } else {                                                                     \
    MLOG(level + 1, fmt, ##__VA_ARGS__)                                        \
  }

#define ABORTIF(cond, msg)                                                     \
  if (cond) {                                                                  \
    LOG(FATAL) << fmtstr(MLOG_BUFSZ, "[ERRO] %s", msg);                        \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  }

#define ABORT(msg)                                                             \
  {                                                                            \
    LOG(FATAL) << fmtstr(MLOG_BUFSZ, "[ERRO] %s", msg);                        \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  }
