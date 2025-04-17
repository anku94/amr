#pragma once

#include <memory>

#include "common.h"

namespace amr {
class ConfigParser;
}

namespace Globals {
extern int my_rank, nranks;
extern DriverOpts driver_opts;
extern std::unique_ptr<amr::ConfigParser> config;
}  // namespace Globals
