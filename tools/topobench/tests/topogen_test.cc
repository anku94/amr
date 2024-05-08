//
// Created by Ankush J on 8/12/22.
//

#include "trace_reader.h"

#include <common.h>
#include <graph.h>
#include <gtest/gtest.h>

// TEST(Topogen_Test, GenerateMesh) {
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

TEST(Topogen_Test, NormalGenerator) {
  int mean = 5; int std = 2;
  int reps = 1000;

  NormalGenerator ng(mean, std);
  int w1sd = 0;

  for (int i = 0; i < reps; i++) {
    double num = ng.GenInt();
    if (num >= (mean - std) and num < (mean + std)) w1sd++;
  }

  double prop_1sd = w1sd * 1.0 / reps;

  logv(__LOG_ARGS__, LOG_INFO, "Normal Generator Test: +-1std: %d/%d", w1sd, reps);

  ASSERT_TRUE(prop_1sd > 0.6 and prop_1sd < 0.8);
}

TEST(Topogen_Test, TraceReader) {
  TraceReader tr("/Users/schwifty/Repos/amr/tools/topobench/tools/msgs/msgs.0.csv");
  tr.Read();
}

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
