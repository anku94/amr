#pragma once

#include <memory>
#include <string>
#include <vector>

#include "amr_opts.h"

namespace amr {
class AMRMonitor;

extern const AMROpts amr_opts;
extern std::unique_ptr<AMRMonitor> monitor;

std::vector<std::string> split_string(const std::string& s, char delimiter);
}  // namespace amr
