//
// Created by Ankush J on 4/17/23.
//

#pragma once

#include "common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace amr {
void Cluster(const std::vector<int>& costlist, std::vector<int>& costlist_new,
             int k, double& mean_rel_error, double& max_rel_error) {
  std::vector<int> costlist_orig(costlist);
  costlist_new.resize(costlist_orig.size());

  int n = costlist_orig.size();
  std::vector<double> centroids(k);
  std::vector<int> assignment(n);

  // Initialize centroids randomly
  std::shuffle(costlist_orig.begin(), costlist_orig.end(),
               std::mt19937(std::random_device()()));
  for (int i = 0; i < k; ++i) {
    centroids[i] = costlist_orig[i];
  }

  // Run k-means algorithm
  bool changed = true;
  while (changed) {
    changed = false;

    // Assign each point to nearest centroid
    for (int i = 0; i < n; ++i) {
      double min_distance = std::numeric_limits<double>::max();
      int min_index = -1;
      for (int j = 0; j < k; ++j) {
        double distance = std::abs(costlist_orig[i] - centroids[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_index = j;
        }
      }
      if (assignment[i] != min_index) {
        assignment[i] = min_index;
        changed = true;
      }
    }

    // Update centroids
    std::fill(centroids.begin(), centroids.end(), 0.0);
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
      int j = assignment[i];
      centroids[j] += costlist_orig[i];
      counts[j]++;
    }
    for (int j = 0; j < k; ++j) {
      if (counts[j] > 0) {
        centroids[j] /= counts[j];
      }
    }
  }

  // Replace each element of costlist_orig with cluster mean in costlist_new
  for (int i = 0; i < n; ++i) {
    int j = assignment[i];
    costlist_new[i] = centroids[j];
  }

  // Calculate mean-squared error
  double mse = 0.0;
  max_rel_error = 0.0;
  for (int i = 0; i < n; ++i) {
    double delta = std::abs(costlist_new[i] - costlist_orig[i]);
    mse += std::pow(delta, 2);
    double cur_rel_error = delta / costlist_orig[i];
    max_rel_error = std::max(max_rel_error, cur_rel_error);
  }

  mse /= n;

  double mse_sqrt = std::pow(mse, 0.5);
  double costlist_avg =
      std::accumulate(costlist.begin(), costlist.end(), 0.0) / costlist.size();
  mean_rel_error = mse_sqrt / costlist_avg;
  logf(LOG_DBUG, "K: %d, mean_rel_error: %.1f%%, max_rel_error: %.1f%%", k,
       mean_rel_error * 100, max_rel_error * 100);
}
}  // namespace amr
