//
// Created by Ankush J on 8/12/22.
//

#include "graph.h"

#include "common.h"

#include <assert.h>
#include <limits.h>
#include <numeric>

void GraphGenerator::GenerateDynamic(int nnodes, int avg_deg) {
  int nedge_total = nnodes * avg_deg / 2;
  LeastConnectedGraph g(nnodes);

  for (int eidx = 0; eidx < nedge_total; eidx++) {
    bool ret = g.AddEdge();
    if (!ret) return;
  }

  g.PrintConnectivityStats();

  return;
}

bool Graph::SanityCheck() {
#define FAIL_IF(x, msg)   \
  if (x) {                \
    logv(__LOG_ARGS__, LOG_WARN, msg);  \
    check_passed = false; \
    goto check_completed; \
  }

  bool check_passed = true;

  int edgevec_sum = std::accumulate(nedgevec_.cbegin(), nedgevec_.cend(), 0);
  FAIL_IF(edgevec_sum != nedges_, "mismatch: edgevec_sum and nedges");

  for (int u = 0; u < nnodes_; u++) {
    FAIL_IF(graph_[u][u], "self-edge not valid");
    FAIL_IF(nedgevec_[u] >= nnodes_, "too many edges");

    int u_nbrcnt = 0;
    for (int v = 0; v < nnodes_; v++) {
      if (graph_[u][v]) u_nbrcnt++;
    }

    FAIL_IF(u_nbrcnt != nedgevec_[u], "mismatch: nedgevec and u_ubrcnt");
  }

#undef FAIL_IF
check_completed:
  if (check_passed) {
    logv(__LOG_ARGS__, LOG_DBUG, "Graph sanity check passed!");
  }

  return check_passed;
}

bool Graph::AddEdge(int u, int v) {
  if (graph_[u][v] == true or graph_[v][u] == true) {
    logv(__LOG_ARGS__, LOG_DBUG, "%d and %d are already connected", u, v);
    return false;
  }

  graph_[u][v] = true;
  graph_[v][u] = true;
  nedges_++;
  nedgevec_[u]++;
  nedgevec_[v]++;

  return true;
}

bool LeastConnectedGraph::AddEdge() {
  assert(nnodes_ >= 2);

  PQNode u = *(least_conn_.begin());
  PQNode v{-1, -1};

  /* Which node to connect with?
   * Unless graph is fully connected, there must be a v s.t. graph_[u][v] =
   * false But v needn't be the next node in least_conn_
   */
  for (auto it = least_conn_.begin(); it != least_conn_.end(); it++) {
    v = *it;
    if (u.second == v.second) continue;

    if (!graph_[u.second][v.second]) {
      least_conn_.erase(least_conn_.begin());
      least_conn_.erase(it);
      break;
    }
  }

  if (v.second == -1) {
    logv(__LOG_ARGS__, LOG_ERRO, "No viable node found!!");
    return false;
  }

  logv(__LOG_ARGS__, LOG_DBG2, "Least connected nodes: %d and %d (edgecnt: %d and %d)",
       u.second, v.second, u.first, v.first);

  bool add_ret = Graph::AddEdge(u.second, v.second);

  if (!add_ret) {
    logv(__LOG_ARGS__, LOG_DBUG, "Edge add failed! Graph must be full.");
  }

  // Bump u and v's connectivity
  u.first++;
  v.first++;

  least_conn_.insert(u);
  least_conn_.insert(v);

  return add_ret;
}

void LeastConnectedGraph::PrintConnectivityStats() const {
  int sum_x = 0, sum_x2 = 0;
  int min_x = INT_MAX;
  int max_x = 0;

  for (int x : nedgevec_) {
    sum_x += x;
    sum_x2 += x * x;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
  }

  int nedges_compl = nnodes_ * (nnodes_ - 1);
  float e_x2 = sum_x2 * 1.0f / nnodes_;
  float e_x = sum_x * 1.0f / nnodes_;
  float nedge_var = e_x2 - e_x * e_x;
  float nedge_std = std::pow(nedge_var, 0.5);

  logv(__LOG_ARGS__, LOG_INFO, "Nodes: %d, Edges: %d (Max: %d, %%Full: %.2f%%)", nnodes_,
       sum_x, nedges_compl, sum_x * 100.0 / nedges_compl);

  logv(__LOG_ARGS__, LOG_INFO, "Edge Count, Max/Max: %d/%d (Mean += std: %.1f += %.1f)",
       min_x, max_x, sum_x * 1.0 / nnodes_, nedge_std);
}
