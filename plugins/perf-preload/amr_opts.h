#include "logging.h"

#include <cstdlib>

namespace amr {
struct AMROpts {
  int print_topk;
  int p2p_enable_matrix_reduce;
  int p2p_enable_matrix_put;
  int rankwise_enabled;
  int tswise_enabled;
  int tracing_enabled;
  std::string output_dir;
  std::string rankwise_fpath;
};

class AMROptUtils {
 private:
  constexpr static const char* kRankwiseOutputFilename = "amrmon_rankwise.txt";
  constexpr static const char* kTraceOutputFmt = "amrmon_trace_%d.txt";
  constexpr static const char* kTswiseOutputFmt = "amrmon_tswise_%d.txt";

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
    AMR_OPT(tswise_enabled, "AMRMON_TSWISE_ENABLED", 0);
    AMR_OPT(tracing_enabled, "AMRMON_TRACING_ENABLED", 0);

    AMR_OPTSTR(output_dir, "AMRMON_OUTPUT_DIR", "/tmp");

    opts.rankwise_fpath = opts.output_dir + "/" + kRankwiseOutputFilename;

    return opts;
  }

  static void LogOpts(const AMROpts& opts) {
    logv(__LOG_ARGS__, LOG_INFO, "AMRMON options:");
    // replace by %30s: %d variants of the parameter logging

    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_PRINT_TOPK", opts.print_topk);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_P2P_ENABLE_REDUCE",
         opts.p2p_enable_matrix_reduce);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_P2P_ENABLE_PUT",
        opts.p2p_enable_matrix_put);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_RANKWISE_ENABLED",
         opts.rankwise_enabled);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_TSWISE_ENABLED",
        opts.tswise_enabled);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %d", "AMRMON_TRACING_ENABLED",
        opts.tracing_enabled);
    logv(__LOG_ARGS__, LOG_INFO, "%30s: %s", "AMRMON_OUTPUT_DIR",
        opts.output_dir.c_str());

    if (opts.rankwise_enabled) {
      logv(__LOG_ARGS__, LOG_INFO, "%30s: %s", "AMRMON_RANKWISE_FPATH",
          opts.rankwise_fpath.c_str());
    }
  }

  static pdlfs::WritableFile* GetTswiseOutputFile(const AMROpts& opts,
                                                  pdlfs::Env* env, int rank) {
    if (!opts.tswise_enabled) {
      return nullptr;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), kTswiseOutputFmt, rank);
    std::string fpath = opts.output_dir + "/" + buf;
    logv(__LOG_ARGS__, LOG_DBUG, "Tswise output fpath: %s", fpath.c_str());

    pdlfs::WritableFile* f;
    pdlfs::Status s = env->NewWritableFile(fpath.c_str(), &f);
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_WARN, "Failed to open file %s", fpath.c_str());
      return nullptr;
    }

    return f;
  }

  static pdlfs::WritableFile* GetTracerOutputFile(const AMROpts& opts,
                                                  pdlfs::Env* env, int rank) {
    if (!opts.tracing_enabled) {
      return nullptr;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), kTraceOutputFmt, rank);

    std::string fpath = opts.output_dir + "/" + buf;
    logv(__LOG_ARGS__, LOG_DBUG, "Tracer output fpath: %s", fpath.c_str());

    pdlfs::WritableFile* f;
    pdlfs::Status s = env->NewWritableFile(fpath.c_str(), &f);
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_WARN, "Failed to open file %s",
           opts.rankwise_fpath.c_str());
      return nullptr;
    }

    return f;
  }
};
}  // namespace amr
