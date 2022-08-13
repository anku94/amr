//
// Created by Ankush J on 8/12/22.
//

#include <gtest/gtest.h>
#include <graph.h>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

//TEST(Topogen_Test, GenerateMesh) {
//  Globals::nranks = 512;
//  Globals::my_rank = 0;
//
//  DriverOpts opts;
//  opts.topology = NeighborTopology::Dynamic;
//  opts.topology_nbrcnt = 30;
//
//  Mesh mesh;
//
//  Status s = Topology::GenerateMesh(opts, mesh);
//  ASSERT_EQ(s, Status::OK);
//}

TEST(Topogen_Test, LeastConnectedGraph) {
  LeastConnectedGraph g(4);
  ASSERT_TRUE(g.AddEdge());
  ASSERT_TRUE(g.AddEdge());
  ASSERT_TRUE(g.AddEdge());
  ASSERT_TRUE(g.AddEdge());
  ASSERT_TRUE(g.AddEdge());
  ASSERT_TRUE(g.AddEdge());
  ASSERT_FALSE(g.AddEdge());
  g.PrintConnectivityStats();
  GraphGenerator::GenerateDynamic(512, 30);
}
