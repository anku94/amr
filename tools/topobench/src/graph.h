//
// Created by Ankush J on 8/12/22.
//

#pragma once

#include <set>
#include <vector>

class GraphGenerator {
 public:
  /* currently generates uniform outdegree; TODO non-uniform out-degree */
  static void GenerateDynamic(int nnodes, int avg_deg);
};

class Graph {
 public:
  explicit Graph(int nnodes) : nnodes_(nnodes), nedges_(0) {
    graph_.resize(nnodes_);

    for (auto it = graph_.begin(); it != graph_.end(); it++) {
      (*it).resize(nnodes_, false);
    }

    nedgevec_.resize(nnodes_, 0);
  }

 protected:
  bool AddEdge(int u, int v);

  bool SanityCheck();

  const int nnodes_;
  int nedges_;
  std::vector<int> nedgevec_;

  typedef std::vector<std::vector<bool>> GraphMat;
  GraphMat graph_;
};

class LeastConnectedGraph : public Graph {
 public:
  explicit LeastConnectedGraph(int nnodes) : Graph(nnodes) {
    for (int n = 0; n < nnodes_; n++) {
      least_conn_.insert({0, n});
    }
  }

  bool AddEdge();

  void PrintConnectivityStats() const;

  typedef std::pair<int, int> PQNode;
  std::set<PQNode> least_conn_;
};