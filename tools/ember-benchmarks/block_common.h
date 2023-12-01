//
// Created by Ankush J on 11/30/23.
//

#pragma once

#include <vector>

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
      int rank = PositionUtils::GetRank(bounds_, neighbor);
      neighbors.push_back(rank);
    }
    return neighbors;
  }

  const Triplet my_;
  const Triplet bounds_;
  const std::vector<Triplet> deltas_face_ = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const std::vector<Triplet> deltas_edge_ = {
      {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
      {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
      {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1}};
  const std::vector<Triplet> deltas_vertex_ = {
      {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
      {1, -1, -1},  {1, -1, 1},  {1, 1, -1},  {1, 1, 1}};
};
