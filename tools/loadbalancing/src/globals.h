#pragma once

#include <memory>
#include "config_parser.h"

namespace amr {
  struct GlobalConfig {
    std::unique_ptr<ConfigParser> config;
  };

  extern GlobalConfig Globals;
}

namespace Globals {
extern int my_rank, nranks;
extern DriverOpts driver_opts;
}; // namespace Globals
