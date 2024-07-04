#pragma once

#include "common.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

namespace amr {
class ConfigParser {
 public:
   explicit ConfigParser() {}

  explicit ConfigParser(const std::string file_path) {
    logv(__LOG_ARGS__, LOG_WARN, "Deprecated: need to use Make() instead\n");
    Initialize(file_path.c_str());
  }

  std::string Strip(const std::string& str) const {
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char ch) {
        return std::isspace(ch);
    });

    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char ch) {
        return std::isspace(ch);
    }).base();

    return (start < end) ? std::string(start, end) : "";
  }

  template <typename T>
  T GetParamOrDefault(const std::string& key, const T& default_val) {
    auto it = params_.find(key);
    if (it == params_.end()) {
      logv(__LOG_ARGS__, LOG_DBG2, "Key %s not found in config file, using default value\n",
           key.c_str());
      return default_val;
    }
    return Convert<T>(it->second);
  }

 private:
  void Initialize(const char* file_path) {
    if (file_path == nullptr) {
      logv(__LOG_ARGS__, LOG_WARN, "Config file not specified\n");
      return;
    }

    if (!std::ifstream(file_path)) {
      // file path is optional, but do not specify an invalid one
      logv(__LOG_ARGS__, LOG_ERRO, "Config file %s does not exist\n", file_path);
      ABORT("Config file does not exist");
      return;
    }

    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      auto delimiter_pos = line.find('=');
      if (delimiter_pos != std::string::npos) {
        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);
        key = Strip(key);
        value = Strip(value);
        params_[key] = value;
      }
    }
  }

  std::unordered_map<std::string, std::string> params_;

  template <typename T>
  T Convert(const std::string& s) {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
};

template<>
double ConfigParser::Convert<double>(const std::string& s);

template<>
int ConfigParser::Convert<int>(const std::string& s);

template<>
std::string ConfigParser::Convert<std::string>(const std::string& s);

}  // namespace amr
