#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh_utils.h"

namespace amr {
// Core AMR mesh class
class Mesh {
 public:
  using LeafSet = std::unordered_set<Loc, LocHash>;
  using LocToIdMap = std::unordered_map<Loc, int, LocHash>;
  using OrderedBlockVec = std::vector<int>;

  struct OrderedMeshNode {
    OrderedBlockVec face, edge, vertex;
  };

  struct OrderedMesh {
    int nblocks;
    std::vector<OrderedMeshNode> nbrmap;
  };

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
    LocToIdMap idmap;
    idmap.reserve(leaves_.size());
    int next_id = 0;

    int64_t root_extent = (1LL << root_level_);
    for (int64_t z = 0; z < root_extent; ++z) {
      for (int64_t y = 0; y < root_extent; ++y) {
        for (int64_t x = 0; x < root_extent; ++x) {
          Loc root_loc{root_level_, {x, y, z}};
          // Only traverse active roots
          if (active_.count(root_loc)) {
            DFSAssign(root_loc, idmap, next_id);
          }
        }
      }
    }

    OrderedMesh om;
    om.nblocks = next_id;
    om.nbrmap.assign(next_id, {});

    // Build neighbor lists for each block with ID
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
      Vec3ll nlocv = (loc.locv << 1);
      nlocv.x |= (i & 1);
      nlocv.y |= ((i >> 1) & 1);
      nlocv.z |= ((i >> 2) & 1);
      children.push_back({nl, nlocv});
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
          Loc l{root_level_, {x, y, z}};
          leaves_.insert(l);
          active_.insert(l);
        }
  }

  // Depth-first assign IDs to leaves
  void DFSAssign(const Loc& loc, LocToIdMap& idmap, int& next_id) const {
    // Check if node is active
    if (!active_.count(loc)) {
      return;
    }

    // Assign ID
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
    out.clear();
    Loc relevant_node = FindRelevantNode(tgt);
    if (!relevant_node.IsValid()) {
      return;
    }

    if (IsLeaf(relevant_node)) {
      out.push_back(relevant_node);
    } else {
      CollectDescendants(relevant_node, out);
    }
  }

  // Predefined neighbor offset sets
  static const std::array<Vec3ll, 6> kFaceOffsets_;
  static const std::array<Vec3ll, 12> kEdgeOffsets_;
  static const std::array<Vec3ll, 8> kVertOffsets_;

  void BuildNeighbors(const Loc& loc, int id, const LocToIdMap& idmap,
                      OrderedMesh& om) const {
    auto& nbrmap_forid = om.nbrmap[id];
    nbrmap_forid.face.clear();
    nbrmap_forid.edge.clear();
    nbrmap_forid.vertex.clear();

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
  void ProcessOffset(const Loc& loc, int id, const LocToIdMap& idmap,
                     OrderedBlockVec& nbrs, const Vec3ll& off) const {
    Loc tgt{loc.level, loc.locv + off};
    std::vector<Loc> covering_leaves;
    GatherLeaves(tgt, covering_leaves);

    for (auto& nl : covering_leaves) {
      auto it = idmap.find(nl);
      if (it != idmap.end() && it->second != id && AreNeighbors(loc, nl)) {
        nbrs.push_back(it->second);
      }
    }

    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  void CollectDescendants(const Loc& loc, std::vector<Loc>& out) const {
    if (!active_.count(loc)) {
      return;
    }

    if (IsLeaf(loc)) {
      out.push_back(loc);
    } else {
      for (const auto& c : GetChildLocs(loc)) {
        CollectDescendants(c, out);
      }
    }
  }

  Loc FindRelevantNode(const Loc& target_in) const {
    Loc target = target_in;
    int64_t max_coord_at_lvl = (1LL << target.level);

    // 1. Handle boundaries
    if (target.locv.AnyLT(0) || target.locv.AnyGT(max_coord_at_lvl)) {
      return Loc();
    }

    Loc current_node;
    if (target.level < root_level_) {  // target above root level
      return Loc();
    }

    // Find the root-level node that covers the target
    current_node.level = root_level_;
    current_node.locv = target.locv >> (target.level - root_level_);

    // Check if this root-level block exists
    if (!active_.count(current_node)) {
      // This part of the domain wasn't initialized or has been fully derefined
      // Search upwards from target's parent to find the coarse block covering
      // it.
      Loc parent = target.Parent();
      while (parent.IsValid() && !active_.count(parent)) {
        parent = parent.Parent();
      }
      if (parent.IsValid() && IsLeaf(parent))
        return parent;  // Found coarse ancestor leaf
      return Loc();     // No covering block found
    }

    // 3. --- Traverse Downwards ---
    for (int lvl = root_level_; lvl < target.level; ++lvl) {
      if (IsLeaf(current_node)) {
        return current_node;  // Hit a leaf before target level (coarser
                              // neighbor)
      }

      // Determine child index for the *next* level (lvl+1) based on target
      // coords
      int shift = target.level - (lvl + 1);
      int child_idx = (((target.locv.x >> shift) & 1LL)) |
                      (((target.locv.y >> shift) & 1LL) << 1) |
                      (((target.locv.z >> shift) & 1LL) << 2);

      // Calculate the location of the next node in the path
      Loc next_node = {
          lvl + 1, (current_node.locv << 1)};  // Level up, shift parent coords
      next_node.locv.x |= (child_idx & 1);
      next_node.locv.y |= ((child_idx >> 1) & 1);
      next_node.locv.z |= ((child_idx >> 2) & 1);

      if (!active_.count(next_node)) {
        // Path terminates here, the block covering the target must be
        // current_node Since the loop condition passed, current_node cannot be
        // a leaf here. This implies an inconsistency OR the target is in an
        // empty region *below* a leaf. Safest return is the last known valid
        // covering node.
        assert(IsLeaf(current_node.Parent()) ||
               current_node.level ==
                   root_level_);  // Parent should exist or be root
        // If traversal failed, the coarser node `current_node` should cover it,
        // but we already checked `IsLeaf(current_node)` at the start of the
        // loop. This suggests the target path leads to an inactive region below
        // an internal node. The technically correct covering block might be the
        // *parent* leaf if one exists upwards. However, returning current_node
        // might be acceptable depending on definition. Let's return the last
        // known *active* node.
        return current_node;  // Return the node before the inactive step.
      }
      current_node = next_node;  // Move to the next level
    }

    // 4. --- Reached Target Level ---
    // The loop finished, current_node is at the target level.
    assert(current_node.level == target.level);
    assert(active_.count(current_node) &&
           "Node at target level must be active if loop completed");
    return current_node;  // Return the node foun
  }

  bool AreNeighbors(const Loc& loc1, const Loc& loc2) const {
    int max_level = std::max(loc1.level, loc2.level);
    int diff1 = max_level - loc1.level;
    int diff2 = max_level - loc2.level;

    auto min1 = loc1.locv << diff1;
    auto min2 = loc2.locv << diff2;

    Vec3ll unit_vec{1, 1, 1};
    Vec3ll size1 = unit_vec << diff1;
    Vec3ll size2 = unit_vec << diff2;

    bool x_touch = (min1.x <= min2.x + size2.x) && (min2.x <= min1.x + size1.x);
    bool y_touch = (min1.y <= min2.y + size2.y) && (min2.y <= min1.y + size1.y);
    bool z_touch = (min1.z <= min2.z + size2.z) && (min2.z <= min1.z + size1.z);

    return x_touch && y_touch && z_touch;
  }
};
}  // namespace amr