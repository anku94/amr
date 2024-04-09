#include "benchmarks/alias_method.h"
#include "common.h"
#include "globals.h"

#include <string>
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
  int N_min;
  int N_max;
};

struct DistributionOpts {
  Distribution d;
  union {
    GaussianOpts gaussian;
    ExpOpts exp;
    PowerLawOpts powerlaw;
  };
};

class DistributionUtils {
 public:
  static void GenDistributionWithDefaults(Distribution d,
                                          std::vector<double>& costs,
                                          int nblocks) {
    double gaussian_mean =
        Globals.config->GetParamOrDefault<double>("gaussian_mean", 10.0);
    double gaussian_std =
        Globals.config->GetParamOrDefault<double>("gaussian_std", 0.5);

    double exp_lambda =
        Globals.config->GetParamOrDefault<double>("exp_lambda", 1.0);

    double powerlaw_alpha =
        Globals.config->GetParamOrDefault<double>("powerlaw_alpha", -3.0);
    int powerlaw_N_min =
        Globals.config->GetParamOrDefault<int>("powerlaw_N_min", 50);
    int powerlaw_N_max =
        Globals.config->GetParamOrDefault<int>("powerlaw_N_max", 100);

    DistributionOpts opts;
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
      opts.powerlaw.N_min = powerlaw_N_min;
      opts.powerlaw.N_max = powerlaw_N_max;
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

 private:
  static std::string DistributionOptsToString(DistributionOpts& opts) {
    if (opts.d == Distribution::kGaussian) {
      return "Gaussian: mean=" + std::to_string(opts.gaussian.mean) +
             ", std=" + std::to_string(opts.gaussian.std);
    } else if (opts.d == Distribution::kExponential) {
      return "Exponential: lambda=" + std::to_string(opts.exp.lambda);
    } else if (opts.d == Distribution::kPowerLaw) {
      return "PowerLaw: alpha=" + std::to_string(opts.powerlaw.alpha) +
             ", N_min=" + std::to_string(opts.powerlaw.N_min) +
             ", N_max=" + std::to_string(opts.powerlaw.N_max);
    } else {
      return "Unknown";
    }
  }

  static void GenDistribution(Distribution d, std::vector<double>& costs,
                              int nblocks) {
    logf(LOG_INFO, "[GenDistribution] Distribution: %s, nblocks: %d",
         DistributionToString(d).c_str(), nblocks);

    costs.resize(nblocks);

    switch (d) {
      case Distribution::kGaussian:
        GenGaussian(costs, nblocks, 10.0, 0.5);
        break;
      case Distribution::kExponential:
        GenExponential(costs, nblocks, 1);
        break;
      case Distribution::kPowerLaw:
        GenPowerLaw(costs, nblocks, -3.0, 50, 100);
        break;
      default:
        GenUniform(costs, nblocks);
        break;
    }
  }

  static void GenDistribution(DistributionOpts& d, std::vector<double>& costs,
                              int nblocks) {
    logf(LOG_INFO, "[GenDistribution] Distribution: %s, nblocks: %d",
         DistributionOptsToString(d).c_str(), nblocks);

    costs.resize(nblocks);

    switch (d.d) {
      case Distribution::kGaussian:
        GenGaussian(costs, nblocks, d.gaussian.mean, d.gaussian.std);
        break;
      case Distribution::kExponential:
        GenExponential(costs, nblocks, d.exp.lambda);
        break;
      case Distribution::kPowerLaw:
        GenPowerLaw(costs, nblocks, d.powerlaw.alpha, d.powerlaw.N_min,
                    d.powerlaw.N_max);
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
};
}  // namespace amr
