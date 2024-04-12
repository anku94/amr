#pragma once

#include <fstream>
#include <string>
#include <unordered_map>

namespace amr {
class ConfigParser {
 public:
  explicit ConfigParser(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
      auto delimiter_pos = line.find('=');
      if (delimiter_pos != std::string::npos) {
        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);
        params_[key] = value;
      }
    }
  }

  template <typename T>
  T GetParamOrDefault(const std::string& key, const T& default_val) {
    auto it = params_.find(key);
    if (it == params_.end()) {
      return default_val;
    }
    return Convert<T>(it->second);
  }

 private:
  std::unordered_map<std::string, std::string> params_;

  template <typename T>
  T Convert(const std::string& val);

  template <>
  int Convert(const std::string& val) {
    return std::stoi(val);
  }

  template <>
  double Convert(const std::string& val) {
    return std::stod(val);
  }

  template <>
  std::string Convert(const std::string& val) {
    return val;
  }
};
}  // namespace amr
