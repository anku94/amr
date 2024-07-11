#include "config_parser.h"

#include <algorithm>
#include <fstream>

namespace amr {
void ConfigParser::Initialize(const char* file_path) {
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

std::string ConfigParser::Strip(const std::string& str) {
  auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char ch) {
    return std::isspace(ch);
  });

  auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char ch) {
               return std::isspace(ch);
             }).base();

  return (start < end) ? std::string(start, end) : "";
}

// specialization of templated functions
template <>
int ConfigParser::Convert(const std::string& s) {
  return std::stoi(s);
}

template <>
double ConfigParser::Convert(const std::string& s) {
  return std::stod(s);
}

template <>
std::string ConfigParser::Convert(const std::string& s) {
  return s;
}
}  // namespace amr
