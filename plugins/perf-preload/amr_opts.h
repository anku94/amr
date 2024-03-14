#include "logging.h"

#include <cstdlib>

namespace amr {
struct AMROpts {
  int print_topk;
  int p2p_enable_matrix_reduce;
  int p2p_enable_matrix_put;
};

class AMROptUtils {
 public:
  static AMROpts GetOpts() {
    AMROpts opts;

#define AMR_OPT(name, env_var, default_value)    \
  {                                              \
    const char* tmp = getenv(env_var);           \
    if (tmp) {                                   \
      opts.name = std::strtol(tmp, nullptr, 10); \
    } else {                                     \
      opts.name = default_value;                 \
    }                                            \
  }

    AMR_OPT(print_topk, "AMRMON_PRINT_TOPK", 20);
    AMR_OPT(p2p_enable_matrix_reduce, "AMRMON_P2P_ENABLE_REDUCE", 1);
    AMR_OPT(p2p_enable_matrix_put, "AMRMON_P2P_ENABLE_PUT", 1);

    return opts;
  }

  static void LogOpts(const AMROpts& opts) {
    Info(__LOG_ARGS__, "AMRMON options:");
    Info(__LOG_ARGS__, "AMRMON_PRINT_TOPK: \t\t%d", opts.print_topk);
    Info(__LOG_ARGS__, "AMRMON_P2P_ENABLE_REDUCE: \t%d",
         opts.p2p_enable_matrix_reduce);
    Info(__LOG_ARGS__, "AMRMON_P2P_ENABLE_PUT: \t%d",
         opts.p2p_enable_matrix_put);
  }
};
}  // namespace amr
