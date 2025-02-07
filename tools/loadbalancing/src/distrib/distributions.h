#include <algorithm>
#include <string>

#include "alias_method.h"
#include "common.h"
#include "config_parser.h"
//
// Created by Ankush J on 11/9/23.
//

#pragma once

#include <vector>

namespace amr {
enum class Distribution {
  kGaussian,
  kExponential,
  kPowerLaw,
  kFromFile  // read from file for debugging
};

struct GaussianOpts {
  double mean;
  double std;
};

struct ExpOpts {
  double lambda;
};

struct PowerLawOpts {
  double alpha;
};

struct FileOpts {
  const char* fname;
};

struct DistributionOpts {
  Distribution d;
  int N_min;
  int N_max;
  union {
    GaussianOpts gaussian;
    ExpOpts exp;
    PowerLawOpts powerlaw;
    FileOpts file;
  };
};

class DistributionUtils {
 public:
  static DistributionOpts GetConfigOpts() {
    // string literals have a different implementation that does not
    // used std::string.
#define DEFINE_STRLIT_PARAM(name, default_val) \
  const char* name = ConfigUtils::GetParamOrDefault(#name, default_val)

#define DEFINE_PARAM(name, type, default_val) \
  type name = ConfigUtils::GetParamOrDefault<type>(#name, default_val)

    DEFINE_STRLIT_PARAM(distribution, "powerlaw");
    DEFINE_STRLIT_PARAM(file_path, "/some/val");

    DEFINE_PARAM(N_min, double, 50);
    DEFINE_PARAM(N_max, double, 100);
    DEFINE_PARAM(gaussian_mean, double, 10.0);
    DEFINE_PARAM(gaussian_std, double, 0.5);
    DEFINE_PARAM(exp_lambda, double, 0.1);
    DEFINE_PARAM(powerlaw_alpha, double, -3.0);

#undef DEFINE_PARAM

    Distribution d = StringToDistribution(distribution);

    DistributionOpts opts;
    opts.N_min = N_min;
    opts.N_max = N_max;

    if (d == Distribution::kGaussian) {
      opts.d = Distribution::kGaussian;
      opts.gaussian.mean = gaussian_mean;
      opts.gaussian.std = gaussian_std;
    } else if (d == Distribution::kExponential) {
      opts.d = Distribution::kExponential;
      opts.exp.lambda = exp_lambda;
    } else if (d == Distribution::kPowerLaw) {
      opts.d = Distribution::kPowerLaw;
      opts.powerlaw.alpha = powerlaw_alpha;
    } else if (d == Distribution::kFromFile) {
      opts.d = Distribution::kFromFile;
      opts.file.fname = file_path;
    } else {
      ABORT("Unknown distribution");
    }

    return opts;
  }

  static std::string DistributionToString(Distribution d) {
    switch (d) {
      case Distribution::kGaussian:
        return "Gaussian";
      case Distribution::kExponential:
        return "Exponential";
      case Distribution::kPowerLaw:
        return "PowerLaw";
      case Distribution::kFromFile:
        return "FromFile";
      default:
        return "Uniform";
    }
  }

  // private:
  static Distribution StringToDistribution(const std::string& s) {
    std::string s_lower = s;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);

    if (s_lower == "gaussian") {
      return Distribution::kGaussian;
    } else if (s_lower == "exponential") {
      return Distribution::kExponential;
    } else if (s_lower == "powerlaw") {
      return Distribution::kPowerLaw;
    } else if (s_lower == "fromfile") {
      return Distribution::kFromFile;
    } else {
      ABORT("Unknown distribution");
    }

    return Distribution::kGaussian;
  }

  static std::string DistributionOptsToString(DistributionOpts& opts) {
    std::string ret;
    if (opts.d == Distribution::kGaussian) {
      ret = "Gaussian: mean=" + std::to_string(opts.gaussian.mean) +
            ", std=" + std::to_string(opts.gaussian.std);
    } else if (opts.d == Distribution::kExponential) {
      ret = "Exponential: lambda=" + std::to_string(opts.exp.lambda);
    } else if (opts.d == Distribution::kPowerLaw) {
      ret = "PowerLaw: alpha=" + std::to_string(opts.powerlaw.alpha);
    } else if (opts.d == Distribution::kFromFile) {
      ret = "FromFile: " + std::string(opts.file.fname);
    } else {
      ret = "unknown";
    }

    if (ret != "unknown") {
      ret += "[N_min: " + std::to_string(opts.N_min) +
             ", N_max: " + std::to_string(opts.N_max) + "]";
    }

    return ret;
  }

  static void GenDistribution(DistributionOpts& d, std::vector<double>& costs,
                              int nblocks) {
    logv(__LOG_ARGS__, LOG_INFO,
         "[GenDistribution] Distribution: %s, nblocks: %d",
         DistributionOptsToString(d).c_str(), nblocks);

    costs.resize(nblocks);

    switch (d.d) {
      case Distribution::kGaussian:
        GenGaussian(costs, nblocks, d.gaussian.mean, d.gaussian.std, d.N_min,
                    d.N_max);
        break;
      case Distribution::kExponential:
        GenExponential(costs, nblocks, d.exp.lambda, d.N_min, d.N_max);
        break;
      case Distribution::kPowerLaw:
        GenPowerLaw(costs, nblocks, d.powerlaw.alpha, d.N_min, d.N_max);
        break;
      case Distribution::kFromFile:
        GenFromFile(costs, nblocks, d.file.fname);
        break;
      default:
        GenUniform(costs, nblocks);
        break;
    }
  }

  static void GenUniform(std::vector<double>& costs, int nblocks) {
    costs.resize(nblocks);
    for (int i = 0; i < nblocks; i++) {
      costs[i] = 1;
    }
  }

  // Generate a gaussian distribution
  static void GenGaussian(std::vector<double>& costs, int nblocks, double mean,
                          double std) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, std);

    for (int i = 0; i < nblocks; i++) {
      costs[i] = d(gen);
    }
  }

  static void GenGaussian(std::vector<double>& costs, int nblocks, double mean,
                          double std, int N_min, int N_max) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(0, 1);

    int N = N_max - N_min + 1;
    std::vector<double> prob(N);
    for (int i = 0; i < N; i++) {
      double rel_std = (i + N_min - mean) / std;
      prob[i] = exp(-pow(rel_std, 2) / 2);
    }

    amr::AliasMethod alias(prob);
    for (int i = 0; i < nblocks; i++) {
      double a = d(gen);
      double b = d(gen);
      int alias_sample = alias.Sample(a, b);
      alias_sample += N_min;
      costs[i] = alias_sample;
    }
  }

  static void GenExponential(std::vector<double>& costs, int nblocks,
                             double lambda) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> d(lambda);

    for (int i = 0; i < nblocks; i++) {
      costs[i] = d(gen);
    }
  }

  static void GenExponentialBuggy(std::vector<double>& costs, int nblocks,
                                  double lambda, int N_min, int N_max) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(0, 1);

    int N = N_max - N_min + 1;
    double norm_factor = 0.0;

    std::vector<double> prob(N);
    for (int i = 0; i < N; i++) {
      double lambda_exp = exp(-lambda * i);
      prob[i] = lambda * lambda_exp;
      norm_factor += prob[i];
    }

    for (int i = 0; i < N; i++) {
      prob[i] /= norm_factor;
    }

    amr::AliasMethod alias(prob);
    for (int i = 0; i < nblocks; i++) {
      double a = d(gen);
      double b = d(gen);
      int alias_sample = alias.Sample(a, b);
      alias_sample += N_min;
      costs[i] = alias_sample;
    }
  }

  static void GenExponential(std::vector<double>& costs, int nblocks,
                             double lambda, int N_min, int N_max) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(1e-10, 1.0);  // Avoid log(0)

    for (int i = 0; i < nblocks; i++) {
      double S = -std::log(d(gen)) / lambda;   // Exponential sample
      int cost = N_min + static_cast<int>(S);  // Shift by N_min
      costs[i] = std::min(cost, N_max);        // Clamp to N_max
    }
  }

  static void GenPowerLaw(std::vector<double>& costs, int nblocks, double alpha,
                          int N_min, int N_max) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(0, 1);

    for (int i = 0; i < nblocks; i++) {
      costs[i] = pow(d(gen), -1.0 / alpha);
    }

    int N = N_max - N_min + 1;
    std::vector<double> prob(N);
    for (int i = 0; i < N; i++) {
      prob[i] = pow(i + N_min, alpha);
    }

    amr::AliasMethod alias(prob);
    for (int i = 0; i < nblocks; i++) {
      double a = d(gen);
      double b = d(gen);
      int alias_sample = alias.Sample(a, b);
      alias_sample += N_min;
      costs[i] = alias_sample;
    }
  }

  static void GenFromFile(std::vector<double>& costs, int nblocks,
                          const char* fname) {
    // file is a space-separated list of unknown number of doubles, read them
    // into costs
    costs.resize(nblocks, 0);

    FILE* f = fopen(fname, "r");
    if (!f) {
      logv(__LOG_ARGS__, LOG_ERRO, "Could not open file: %s", fname);
    }

    for (int i = 0; i < nblocks; i++) {
      int rv = fscanf(f, "%lf", &costs[i]);
      if (rv == EOF) {
        logv(__LOG_ARGS__, LOG_WARN, "EOF reached at %d blocks", i);
        break;
      }
    }

    fclose(f);
  }

  static std::string PlotHistogram(std::vector<double> const& costs, int N_min,
                                   int N_max, int max_height) {
    std::vector<int> hist(N_max - N_min + 1, 0);
    for (auto c : costs) {
      int idx = c - N_min;
      hist[idx]++;
    }

    int max_count = *std::max_element(hist.begin(), hist.end());
    int scale = max_count / max_height;

    std::stringstream ss;
    for (int i = 0; i < hist.size(); i++) {
      ss << "[" << i + N_min << "] ";
      for (int j = 0; j < hist[i] / scale; j++) {
        ss << "*";
      }
      ss << std::endl;
    }

    return ss.str();
  }
};
}  // namespace amr
