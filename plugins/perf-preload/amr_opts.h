#include "logging.h"

#include <cstdlib>

namespace amr {
struct AMROpts {
  int print_topk;
  int p2p_enable_matrix_reduce;
  int p2p_enable_matrix_put;
  int rankwise_enabled;
  std::string output_dir;
  std::string rankwise_fpath;
};

class AMROptUtils {
 private:
  constexpr static const char* kRankwiseOutputFilename = "amrmon_rankwise.txt";

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

#define AMR_OPTSTR(name, env_var, default_value) \
  {                                              \
    const char* tmp = getenv(env_var);           \
    if (tmp) {                                   \
      opts.name = tmp;                           \
    } else {                                     \
      opts.name = default_value;                 \
    }                                            \
  }

    AMR_OPT(print_topk, "AMRMON_PRINT_TOPK", 40);
    AMR_OPT(p2p_enable_matrix_reduce, "AMRMON_P2P_ENABLE_REDUCE", 1);
    AMR_OPT(p2p_enable_matrix_put, "AMRMON_P2P_ENABLE_PUT", 1);
    AMR_OPT(rankwise_enabled, "AMRMON_RANKWISE_ENABLED", 0);

    AMR_OPTSTR(output_dir, "AMRMON_OUTPUT_DIR", "/tmp");

    opts.rankwise_fpath = opts.output_dir + "/" + kRankwiseOutputFilename;

    return opts;
  }

  static void LogOpts(const AMROpts& opts) {
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON options:");
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_PRINT_TOPK: \t\t%d", opts.print_topk);
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_P2P_ENABLE_REDUCE: \t%d",
         opts.p2p_enable_matrix_reduce);
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_P2P_ENABLE_PUT: \t%d",
         opts.p2p_enable_matrix_put);
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_RANKWISE_ENABLED: \t%d", opts.rankwise_enabled);
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_OUTPUT_DIR: \t\t%s", opts.output_dir.c_str());
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON_RANKWISE_FPATH: \t%s",
         opts.rankwise_fpath.c_str());
  }
};
}  // namespace amr
