#pragma once

#include <glog/logging.h>  // IWYU pragma: export

inline std::string fmtstr(const char *fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

#define MLOG_INNER(lvl, fmt, ...)                             \
  do {                                                        \
    switch (lvl) {                                            \
      case LOG_ERRO:                                          \
        LOG(ERROR) << fmtstr("[ERRO] " fmt, ##__VA_ARGS__);   \
        break;                                                \
      case LOG_WARN:                                          \
        LOG(WARNING) << fmtstr("[WARN] " fmt, ##__VA_ARGS__); \
        break;                                                \
      case LOG_INFO:                                          \
        LOG(INFO) << fmtstr("[INFO] " fmt, ##__VA_ARGS__);    \
        break;                                                \
      case LOG_DBUG:                                          \
        VLOG(0) << fmtstr("[DBG0] " fmt, ##__VA_ARGS__);      \
        break;                                                \
      case LOG_DBG2:                                          \
        VLOG(2) << fmtstr("[DBG2] " fmt, ##__VA_ARGS__);      \
        break;                                                \
      case LOG_DBG3:                                          \
        VLOG(3) << fmtstr("[DBG3] " fmt, ##__VA_ARGS__);      \
        break;                                                \
    }                                                         \
  } while (0)

#define MLOG(level, fmt, ...)                                                  \
  MLOG_INNER(level, "[%10.10s:%3.3d] " fmt,                                    \
             (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__), \
             __LINE__, ##__VA_ARGS__)