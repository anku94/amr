//
// Created by Ankush J on 11/30/23.
//

#pragma once

#include <cstdio>
#include <glog/logging.h>
#include <memory>
#include <vector>

class RankMap;
extern std::unique_ptr<RankMap> RANK_MAP;

struct Triplet {
  int x;
  int y;
  int z;

  Triplet() : x(0), y(0), z(0) {}

  Triplet(int x, int y, int z) : x(x), y(y), z(z) {}

  // define a + operator
  Triplet operator+(const Triplet& other) const {
    Triplet result;
    result.x = x + other.x;
    result.y = y + other.y;
    result.z = z + other.z;
    return result;
  }
};

class PositionUtils {
 public:
  static Triplet GetPosition(int rank, Triplet bounds) {
    Triplet my_pos;
    const int plane = rank % (bounds.x * bounds.y);
    my_pos.y = plane / bounds.x;
    my_pos.x = (plane % bounds.x) != 0 ? (plane % bounds.x) : 0;
    my_pos.z = rank / (bounds.x * bounds.y);
    return my_pos;
  }

  static int GetRank(Triplet bounds, Triplet my) {
    if (my.x < 0 or my.y < 0 or my.z < 0) {
      return -1;
    }

    if (my.x >= bounds.x or my.y >= bounds.y or my.z >= bounds.z) {
      return -1;
    }

    int rank = (my.z * bounds.x * bounds.y) + (my.y * bounds.x) + my.x;
    return rank;
  }
};

class RankMap {
 public:
  RankMap(int nranks, const char* map_file)
      : map_file_(map_file), log_to_phy_(nranks, -1), phy_to_log_(nranks, -1) {
    LoadMapFile();
  }

  void LoadMapFile() {
    if (map_file_ == nullptr) {
      return;
    }

    FILE* f = fopen(map_file_, "r");
    int a = -1, b = -1;
    while (fscanf(f, "%d,%d\n", &a, &b) != EOF) {
      LOG_IF(FATAL, a == -1 or b == -1) << "Invalid map file: " << map_file_;
      LOG_IF(FATAL, a >= log_to_phy_.size() or b >= phy_to_log_.size())
          << "[RankMap] a >= asz or b >= bsz";

      phy_to_log_[a] = b;
      log_to_phy_[b] = a;
    }

    fclose(f);
  }

  int GetLogicalRank(int phy_rank) {
    if (map_file_ == nullptr) {
      return phy_rank;
    }

    if (phy_rank == -1) return -1;
    LOG_IF(FATAL, phy_rank >= phy_to_log_.size())
        << "[RankMap] phy_rank >= phy_to_log_.size()";

    return phy_to_log_[phy_rank];
  }

  int GetPhysicalRank(int log_rank) {
    if (map_file_ == nullptr) {
      return log_rank;
    }

    if (log_rank == -1) return -1;
    LOG_IF(FATAL, log_rank >= log_to_phy_.size())
        << "[RankMap] log_rank >= log_to_phy_.size()";

    return log_to_phy_[log_rank];
  }

 private:
  const char* map_file_;
  std::vector<int> log_to_phy_;
  std::vector<int> phy_to_log_;
};

class NeighborRankGenerator {
 public:
  NeighborRankGenerator(const Triplet& my, const Triplet& bounds)
      : my_(my), bounds_(bounds) {}

  std::vector<int> GetFaceNeighbors() const {
    return GetNeighbors(deltas_face_);
  }

  std::vector<int> GetEdgeNeighbors() const {
    return GetNeighbors(deltas_edge_);
  }

  std::vector<int> GetVertexNeighbors() const {
    return GetNeighbors(deltas_vertex_);
  }

 private:
  std::vector<int> GetNeighbors(const std::vector<Triplet>& deltas) const {
    std::vector<int> neighbors;

    for (auto delta : deltas) {
      Triplet neighbor = my_ + delta;
      int log_rank = PositionUtils::GetRank(bounds_, neighbor);
      int rank = RANK_MAP->GetPhysicalRank(log_rank);
      neighbors.push_back(rank);
    }

    return neighbors;
  }

  const Triplet my_;
  const Triplet bounds_;
  const std::vector<Triplet> deltas_face_ = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const std::vector<Triplet> deltas_edge_ = {
      {-1, -1, 0}, {0, -1, -1}, {1, -1, 0}, {0, -1, 1}, {-1, 0, 1}, {1, 0, 1},
      {-1, 0, -1}, {1, 0, -1},  {-1, 1, 0}, {0, 1, 1},  {1, 1, 0},  {0, 1, -1}};
  const std::vector<Triplet> deltas_vertex_ = {
      {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
      {1, -1, -1},  {1, -1, 1},  {1, 1, -1},  {1, 1, 1}};
};
