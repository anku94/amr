//
// Created by Ankush J on 4/11/23.
//

#include "policy_exec_ctx.h"

#include <gtest/gtest.h>

namespace amr {
class MiscTest : public ::testing::Test {
 protected:
  std::string GetLogPath(const char* output_dir, const char* policy_name) {
    return PolicyExecutionContext::GetLogPath(output_dir, policy_name);
  }
};

TEST_F(MiscTest, OutputFileTest) {
  std::string policy_name = "RoundRobin_Actual-Cost";
  std::string fname = GetLogPath("/a/b/c", policy_name.c_str());
  ASSERT_EQ(fname, "/a/b/c/lb_sim_roundrobin_actual_cost.csv");
}
}  // namespace amr