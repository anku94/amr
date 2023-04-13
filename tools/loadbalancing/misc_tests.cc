//
// Created by Ankush J on 4/11/23.
//

#include "policy_exec_ctx.h"

#include <gtest/gtest.h>

namespace amr {
class MiscTest : public ::testing::Test {
 protected:
  std::string GetLogPath(PolicyExecutionContext& ctx) {
    return ctx.GetLogPath();
  }
};

TEST_F(MiscTest, OutputFileTest) {
  pdlfs::Env* env = pdlfs::Env::Default();
  std::string policy_name = "abcd/efgh";
  PolicyExecutionContext ctx(policy_name.c_str(), Policy::kPolicyLPT, env);
  std::string fname = GetLogPath(ctx);
  ASSERT_EQ(fname, "abcd_efgh");
}
}  // namespace amr