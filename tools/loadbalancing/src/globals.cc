#include "globals.h"

#include "config_parser.h"

namespace Globals {
int my_rank, nranks;
DriverOpts driver_opts;
std::unique_ptr<amr::ConfigParser> config;
}  // namespace Globals
