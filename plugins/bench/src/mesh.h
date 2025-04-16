#pragma once

#include "mesh_utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace amr {
// Core AMR mesh class
class Mesh {
 public:
  using LeafSet = std::unordered_set<Loc, LocHash>;
  using LocToIDMap = std::unordered_map<Loc, int, LocHash>;
  using OrderedBlockVec = std::vector<int>;

  struct OrderedMeshNode {
    OrderedBlockVec face, edge, vertex;
  };

  struct OrderedMesh {
    int nblocks;
    std::vector<OrderedMeshNode> nbrmap;
  };

  Mesh(int nx, int ny, int nz, int max_level) : max_level_(max_level) {
    assert(nx > 0 && ny > 0 && nz > 0 && max_level >= 0);
    root_level_ = 0;
    int m = std::max({nx, ny, nz});
    while ((1 << root_level_) < m) ++root_level_;
    assert(max_level_ >= root_level_);
    InitializeRoots(nx, ny, nz);
    current_max_level_ = root_level_;
  }

  int GetRootLevel() const { return root_level_; }
  int GetCurrentMaxLevel() const { return current_max_level_; }
  bool IsLeaf(const Loc& loc) const { return leaves_.count(loc) != 0; }

  void Refine(const Loc& loc) {
    assert(IsLeaf(loc) && loc.level < max_level_);
    leaves_.erase(loc);
    current_max_level_ = std::max(current_max_level_, loc.level + 1);
    for (auto& c : GetChildLocs(loc)) {
      leaves_.insert(c);
      active_.insert(c);
    }
  }

  void Derefine(const Loc& leaf) {
    assert(IsLeaf(leaf) && leaf.level > root_level_);
    Loc parent = leaf.Parent();
    for (auto& sib : GetChildLocs(parent)) {
      leaves_.erase(sib);
      active_.erase(sib);
    }
    leaves_.insert(parent);
  }

  std::vector<Loc> GetLeafLocations() const {
    return {leaves_.begin(), leaves_.end()};
  }

  OrderedMesh GetOrderedMesh() const {
    // DFS numbering
    LocToIDMap idmap;
    idmap.reserve(leaves_.size());
    int next_id = 0;
    DFSAssign({root_level_, 0, 0, 0}, idmap, next_id);

    OrderedMesh om;
    om.nblocks = next_id;
    om.nbrmap.assign(next_id, {});

    // Build neighbor lists
    for (auto& kv : idmap) {
      BuildNeighbors(kv.first, kv.second, idmap, om);
    }
    return om;
  }

  // Generate direct children of a Loc
  static std::vector<Loc> GetChildLocs(const Loc& loc) {
    std::vector<Loc> children;
    children.reserve(8);
    int nl = loc.level + 1;
    for (int i = 0; i < 8; ++i) {
      children.push_back({nl, (loc.lx << 1) | (i & 1),
                          (loc.ly << 1) | ((i >> 1) & 1),
                          (loc.lz << 1) | ((i >> 2) & 1)});
    }
    return children;
  }

 private:
  int root_level_, max_level_, current_max_level_;
  LeafSet leaves_, active_;

  void InitializeRoots(int nx, int ny, int nz) {
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          Loc l{root_level_, x, y, z};
          leaves_.insert(l);
          active_.insert(l);
        }
  }

  // Depth-first assign IDs to leaves
  void DFSAssign(const Loc& loc, LocToIDMap& idmap, int& next_id) const {
    if (IsLeaf(loc)) {
      idmap[loc] = next_id++;
    } else {
      for (const auto& c : GetChildLocs(loc)) {
        DFSAssign(c, idmap, next_id);
      }
    }
  }

  // Collect all leaf Locs covering target
  void GatherLeaves(const Loc& tgt, std::vector<Loc>& out) const {
    if (IsLeaf(tgt)) {
      out.push_back(tgt);
    } else if (active_.count(tgt)) {
      for (const auto& c : GetChildLocs(tgt)) {
        GatherLeaves(c, out);
      }
    } else if (tgt.IsValid()) {
      GatherLeaves(tgt.Parent(), out);
    }
  }

  // Predefined neighbor offset sets
  static const std::array<Vec3i, 6> kFaceOffsets_;
  static const std::array<Vec3i, 12> kEdgeOffsets_;
  static const std::array<Vec3i, 8> kVertOffsets_;

  void BuildNeighbors(const Loc& loc, int id, const LocToIDMap& idmap,
                      OrderedMesh& om) const {
    auto& nbrmap_forid = om.nbrmap[id];
    for (auto& off : kFaceOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.face, off);
    }

    for (auto& off : kEdgeOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.edge, off);
    }

    for (auto& off : kVertOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.vertex, off);
    }
  }

  // Helper to process one offset array and fill neighbor vector
  void ProcessOffset(const Loc& loc, int id, const LocToIDMap& idmap,
                     OrderedBlockVec& nbrs, const Vec3i& off) const {
    Loc tgt{loc.level, loc.lx + off.x, loc.ly + off.y, loc.lz + off.z};
    std::vector<Loc> cov;
    GatherLeaves(tgt, cov);
    for (auto& nl : cov) {
      auto it = idmap.find(nl);
      if (it != idmap.end()) nbrs.push_back(it->second);
    }
  }
};
}  // namespace amr
