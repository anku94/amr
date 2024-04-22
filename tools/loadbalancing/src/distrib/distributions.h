#include <string>

#include "alias_method.h"
#include "common.h"
#include "globals.h"
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

struct DistributionOpts {
  Distribution d;
  int N_min;
  int N_max;
  union {
    GaussianOpts gaussian;
    ExpOpts exp;
    PowerLawOpts powerlaw;
  };
};

class DistributionUtils {
 public:
  static Distribution GetConfigDistribution() {
    std::string distrib_str = Globals.config->GetParamOrDefault<std::string>(
        "distribution", "powerlaw");
    return StringToDistribution(distrib_str);
  }

  static void GenDistributionWithDefaults(std::vector<double>& costs,
                                          int nblocks) {
    Distribution d = GetConfigDistribution();

    double N_min = Globals.config->GetParamOrDefault<int>("N_min", 50);
    double N_max = Globals.config->GetParamOrDefault<int>("N_max", 100);

    double gaussian_mean =
        Globals.config->GetParamOrDefault<double>("gaussian_mean", 10.0);
    double gaussian_std =
        Globals.config->GetParamOrDefault<double>("gaussian_std", 0.5);

    double exp_lambda =
        Globals.config->GetParamOrDefault<double>("exp_lambda", 0.1);

    double powerlaw_alpha =
        Globals.config->GetParamOrDefault<double>("powerlaw_alpha", -3.0);

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
    } else {
      ABORT("Unknown distribution");
    }

    GenDistribution(opts, costs, nblocks);
  }

  static std::string DistributionToString(Distribution d) {
    switch (d) {
      case Distribution::kGaussian:
        return "Gaussian";
      case Distribution::kExponential:
        return "Exponential";
      case Distribution::kPowerLaw:
        return "PowerLaw";
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

  static void GenExponential(std::vector<double>& costs, int nblocks,
                             double lambda, int N_min, int N_max) {
    costs.resize(nblocks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(0, 1);

    int N = N_max - N_min + 1;
    std::vector<double> prob(N);
    for (int i = 0; i < N; i++) {
      prob[i] = lambda * exp(-lambda * (i + N_min));
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
