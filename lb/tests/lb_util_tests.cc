
#include <gtest/gtest.h>

#include "assignment_cache.h"

namespace amr {
class LBUtilTest : public ::testing::Test {};

void AssertVectorEqual(const std::vector<int>& a, const std::vector<int>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

TEST_F(LBUtilTest, AssignmentCacheTest) {
  AssignmentCache cache(2);

  std::vector<int> ranklist;
  bool rv;

  cache.Put(3, {1, 2, 3});
  rv = cache.Get(3, ranklist);
  ASSERT_TRUE(rv);
  AssertVectorEqual(ranklist, {1, 2, 3});

  cache.Put(4, {1, 2, 3, 4});
  rv = cache.Get(4, ranklist);
  ASSERT_TRUE(rv);

  cache.Put(5, {1, 2, 3, 4, 5});
  rv = cache.Get(5, ranklist);
  rv = cache.Get(3, ranklist);
  ASSERT_TRUE(rv);

  rv = cache.Get(3, ranklist);
  ASSERT_FALSE(rv);
}
}  // namespace amr
