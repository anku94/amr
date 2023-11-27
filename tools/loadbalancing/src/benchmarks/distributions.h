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

class DistributionUtils {
 public:
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