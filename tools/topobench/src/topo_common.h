#pragma once

#include <glog/logging.h> // IWYU pragma: export

#define MLOG(lvl, fmt, ...) \
  logv(__LOG_ARGS__, lvl, fmt, ##__VA_ARGS__)