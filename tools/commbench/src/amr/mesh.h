#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh_utils.h"
#include "tools-common/logging.h"
namespace topo {
enum class NeighborType {
  kFace,
  kEdge,
  kVertex,
  kNone,
};

// Core AMR mesh class
class Mesh {
 public:
  using LeafSet = std::unordered_set<Loc, LocHash>;
  using LocToIdMap = std::unordered_map<Loc, int, LocHash>;


  // Mesh:: ctor
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
  bool IsActive(const Loc& loc) const { return active_.count(loc) != 0; }

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

  OrderedMesh GetOrderedMesh() const;

  // Generate direct children of a Loc
  static std::vector<Loc> GetChildLocs(const Loc& loc);

  // GetTotalBlockCount: total blocks (including intermediate)
  int GetTotalBlockCount() const { return active_.size(); }

  // GetLeafBlockCount: leaf block count only
  int GetLeafBlockCount() const { return leaves_.size(); }

  // GetRootLoc: root location
  static Loc GetRootLoc() { return Loc{0, {0, 0, 0}}; }

  // RefineToTargetLeafcnt: Refine randomly until we reach tgt_leafcnt
  int RefineToTargetLeafcnt(int tgt_leafcnt);


 private:
  int root_level_, max_level_, current_max_level_;
  LeafSet leaves_, active_;

  void InitializeRoots(int nx, int ny, int nz);

  // Depth-first assign IDs to leaves
  void DFSAssign(const Loc& loc, LocToIdMap& idmap, int& next_id) const;

  // Predefined neighbor offset sets
  static const std::array<Vec3ll, 6> kFaceOffsets_;
  static const std::array<Vec3ll, 12> kEdgeOffsets_;
  static const std::array<Vec3ll, 8> kVertOffsets_;

  void BuildNeighbors(const Loc& loc, int id, const LocToIdMap& idmap,
                      OrderedMesh& om) const {
    MLOG(MLOG_DBG0, "--- Building neighbors for block %d ---\n", id);

    auto& nbrmap_forid = om.nbrmap[id];
    nbrmap_forid.face.clear();
    nbrmap_forid.edge.clear();
    nbrmap_forid.vertex.clear();

    for (auto& off : kFaceOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.face, off, NeighborType::kFace);
    }

    for (auto& off : kEdgeOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.edge, off, NeighborType::kEdge);
    }

    for (auto& off : kVertOffsets_) {
      ProcessOffset(loc, id, idmap, nbrmap_forid.vertex, off, NeighborType::kVertex);
    }
  }

  // Helper to process one offset array and fill neighbor vector
  void ProcessOffset(const Loc& loc, int id, const LocToIdMap& idmap,
                     OrderedBlockVec& nbrs, const Vec3ll& off, NeighborType ntype) const;

  NeighborType AreNeighborsByType(const Loc& loc1, const Loc& loc2) const;

  const char* NeighborTypeToString(NeighborType ntype) const {
    switch (ntype) {
      case NeighborType::kFace:
        return "Face";
      case NeighborType::kEdge:
        return "Edge";
      case NeighborType::kVertex:
        return "Vertex";
      case NeighborType::kNone:
        return "None";
      default:
        return "Unknown";
    }
  }

  std::vector<Loc> GetCoveringLeaves(const Loc& t) const;
};
}  // namespace amr
