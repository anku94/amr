#pragma once

#include <memory>
#include <string>
#include <vector>

namespace amr {
class AMRMonitor;

extern std::unique_ptr<AMRMonitor> monitor;

std::vector<std::string> split_string(const std::string& s, char delimiter);
}  // namespace amr
