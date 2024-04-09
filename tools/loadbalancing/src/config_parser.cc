#include "config_parser.h"

namespace amr {
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
