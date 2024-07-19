//
// Created by Ankush J on 7/13/23.
//
#include "common.h"
#include "scaling/scale_sim.h"
#include "distrib/distributions.h"

#include <gtest/gtest.h>

namespace amr {
class ScaleTest : public ::testing::Test {
 public:
 private:
};

TEST_F(ScaleTest, HelloWorld) {
  EXPECT_EQ(1, 1);
  std::vector<double> costs(512, 0.0);
  auto opts = DistributionUtils::GetConfigOpts();
  DistributionUtils::GenDistribution(opts, costs, 512);
}
}  // namespace amr
