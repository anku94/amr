#pragma once

#include <cstdio>
#include <string>

#define LOG_FATAL 0
#define LOG_ERROR 1
#define LOG_WARN 2
#define LOG_INFO 3
#define LOG_DBUG 4
#define LOG_DBG2 5

#define LOG_LEVEL LOG_DBUG

inline std::string LogPrefix(int lvl) {
  switch (lvl) {
    case LOG_DBG2:
      return "[DBG2]";
    case LOG_DBUG:
      return "[DBUG]";
    case LOG_INFO:
      return "[INFO]";
    case LOG_WARN:
      return "[WARN]";
    case LOG_ERROR:
      return "[ERRO]";
    case LOG_FATAL:
      return "[FATL]";
    default:
      return "[UNKN]";
  }
}

inline void logf(int lvl, const char* fmt, ...) {
  if (lvl <= LOG_LEVEL) {
    // Print prefix first
    printf("%s ", LogPrefix(lvl).c_str());

    // Then handle the formatted message
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }
}

#define LOG(lvl, fmt, ...) logf(lvl, fmt, ##__VA_ARGS__)
