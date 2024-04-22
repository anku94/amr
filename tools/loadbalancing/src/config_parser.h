#pragma once

#include "common.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

namespace amr {
class ConfigParser {
 public:
  explicit ConfigParser(const std::string& file_path) {
    // if file does not exist, warn but do nothing
    if (!std::ifstream(file_path)) {
      logv(__LOG_ARGS__, LOG_WARN, "Config file %s does not exist\n", file_path.c_str());
      return;
    }

    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
      // if line begins with #, continue
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
