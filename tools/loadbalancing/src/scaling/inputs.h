//
// Created by Ankush J on 7/13/23.
//

#pragma once

#include "common.h"

#include <algorithm>
#include <random>
#include <vector>

namespace amr {
class Inputs {
 public:
  static void GenerateCosts(std::vector<double>& costs) {
    PopulateStdLogNormal(costs);

    std::string costs_str = SerializeVector(costs, 25);
    logv(__LOG_ARGS__, LOG_INFO, "Costs: %s", costs_str.c_str());
  }

 private:
  static void PopulateStdGaussian(std::vector<double>& data) {
    double mean = 0;
    double std = 1.0;

    PopulateGaussian(data, mean, std);
  }

  static void PopulateGaussian(std::vector<double>& data, double mean,
                               double std) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mean, std};
    std::generate(data.begin(), data.end(), [&]() { return d(gen); });
  }

  static void PopulateStdLogNormal(std::vector<double>& data) {
    const double mu = 3;
    const double sigma = 1.0;

    const double clip_low = 15.0;
    const double clip_hi = 80.0;

    PopulateLogNormal(data, mu, sigma);

    auto clip_fn = [&clip_low, clip_hi](double x) {
      x = std::max(x, clip_low);
      x = std::min(x, clip_hi);
      return x;
    };

    std::transform(data.begin(), data.end(), data.begin(), clip_fn);
  }

  //
  // LogNormals have mu and sigma.
  // More sigma equals more skewedness in the shape.
  // mediam (m) = e^mu = median. mu = log(m).
  //
  static void PopulateLogNormal(std::vector<double>& data, double m, double s) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::lognormal_distribution<> d{m, s};

    std::generate(data.begin(), data.end(), [&]() { return d(gen); });
  }
};
}  // namespace amr
