#include "mesh.h"

#include "tools-common/logging.h"
#include "mesh_utils.h"
#include "print_utils.h"

namespace topo {
// Define the offset arrays
inline const std::array<Vec3ll, 6> Mesh::kFaceOffsets_ = {
    {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};

inline const std::array<Vec3ll, 12> Mesh::kEdgeOffsets_ = {{{1, 1, 0},
                                                            {1, -1, 0},
                                                            {-1, 1, 0},
                                                            {-1, -1, 0},
                                                            {1, 0, 1},
                                                            {1, 0, -1},
                                                            {-1, 0, 1},
                                                            {-1, 0, -1},
                                                            {0, 1, 1},
                                                            {0, 1, -1},
                                                            {0, -1, 1},
                                                            {0, -1, -1}}};

inline const std::array<Vec3ll, 8> Mesh::kVertOffsets_ = {{{1, 1, 1},
                                                           {1, 1, -1},
                                                           {1, -1, 1},
                                                           {1, -1, -1},
                                                           {-1, 1, 1},
                                                           {-1, 1, -1},
                                                           {-1, -1, 1},
                                                           {-1, -1, -1}}};

void Mesh::ProcessOffset(const Loc& loc, int id, const LocToIdMap& idmap,
                         OrderedBlockVec& nbrs, const Vec3ll& off,
                         NeighborType ntype) const {
  Loc tgt{loc.level, loc.locv + off};
  std::vector<Loc> covering_leaves = GetCoveringLeaves(tgt);

  for (auto& nl : covering_leaves) {
    auto it = idmap.find(nl);
    if (it != idmap.end() && it->second != id) {
      bool is_neighbor_by_type = AreNeighborsByType(loc, nl) == ntype;

      MLOG(MLOG_DBG2, "Loc: %s, nbrloc: %s, isntype:%s?::%s\n",
           loc.ToString().c_str(), nl.ToString().c_str(),
           NeighborTypeToString(ntype), is_neighbor_by_type ? "true" : "false");

      if (is_neighbor_by_type) {
        nbrs.push_back(it->second);
      }
    }
  }

  std::sort(nbrs.begin(), nbrs.end());
  nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

  MLOG(MLOG_DBG3, "+ Gathered %zu nbrs: %s\n", nbrs.size(),
       PrintUtils::SerializeVec(nbrs).c_str());
}

OrderedMesh Mesh::GetOrderedMesh() const {
  // DFS numbering
  LocToIdMap idmap;
  idmap.reserve(leaves_.size());
  int next_id = 0;

  for (auto& l : active_) {
    if (l.level == root_level_) {
      DFSAssign(l, idmap, next_id);
    }
  }

  OrderedMesh om;
  om.nblocks = next_id;
  om.nbrmap.assign(next_id, {});

  // Build neighbor lists for each block with ID
  for (auto& kv : idmap) {
    auto block_loc = kv.first;
    int block_id = kv.second;
    BuildNeighbors(block_loc, block_id, idmap, om);
  }
  return om;
}

std::vector<Loc> Mesh::GetChildLocs(const Loc& loc) {
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

int Mesh::RefineToTargetLeafcnt(int tgt_leafcnt) {
  int cur_leafcnt = leaves_.size();
  // Warning: can get stuck in an infinite loop if tgt_leafcnt is too high
  // or if refinement is not possible. Just have generous limits.
  while (cur_leafcnt < tgt_leafcnt) {
    // Refine a random active block
    auto it = leaves_.begin();
    std::advance(it, rand() % leaves_.size());
    auto leaf_loc = *it;
    if (leaf_loc.level == max_level_) {
      continue;
    }

    Refine(leaf_loc);
    cur_leafcnt = leaves_.size();
    MLOG(MLOG_DBG0, "Refined block %s, now have %d leaves",
         leaf_loc.ToString().c_str(), cur_leafcnt);
  }

  return cur_leafcnt;
}

void Mesh::InitializeRoots(int nx, int ny, int nz) {
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        Loc l{root_level_, {x, y, z}};
        leaves_.insert(l);
        active_.insert(l);
      }
}

void Mesh::DFSAssign(const Loc& loc, LocToIdMap& idmap, int& next_id) const {
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

NeighborType Mesh::AreNeighborsByType(const Loc& l1, const Loc& l2) const {
  // 1) Scale both blocks up to the same finest level
  int L = std::max(l1.level, l2.level);
  int d1 = L - l1.level, d2 = L - l2.level;

  // 2) Compute half-open intervals [min, max)
  Vec3ll min1 = l1.locv << d1;
  Vec3ll min2 = l2.locv << d2;
  Vec3ll size1 = Vec3ll{1, 1, 1} << d1;
  Vec3ll size2 = Vec3ll{1, 1, 1} << d2;
  Vec3ll max1 = min1 + size1;
  Vec3ll max2 = min2 + size2;

  // 3) On each axis, check overlap vs touch
  auto axis_info = [&](int a_min1, int a_max1, int a_min2, int a_max2) {
    bool overlap = (a_min1 < a_max2) && (a_min2 < a_max1);
    bool touch = (a_max1 == a_min2) || (a_max2 == a_min1);
    return std::make_pair(overlap, touch);
  };

  auto [ox, tx] = axis_info(min1.x, max1.x, min2.x, max2.x);
  auto [oy, ty] = axis_info(min1.y, max1.y, min2.y, max2.y);
  auto [oz, tz] = axis_info(min1.z, max1.z, min2.z, max2.z);

  // 4) Must overlap or touch on *all* axes
  if (!((ox || tx) && (oy || ty) && (oz || tz))) {
    return NeighborType::kNone;
  }

  // 5) Count how many axes are pure-touch
  int nTouch = int(tx) + int(ty) + int(tz);
  switch (nTouch) {
    case 1:
      return NeighborType::kFace;
    case 2:
      return NeighborType::kEdge;
    case 3:
      return NeighborType::kVertex;
    default:
      // nTouch==0 means all three axes overlap => same block or containment
      return NeighborType::kNone;
  }
}

std::vector<Loc> Mesh::GetCoveringLeaves(const Loc& t) const {
  // 1) find the minimal active ancestor of ‘t’
  Loc node = t;
  while (node.level > root_level_ && !active_.count(node)) {
    node = node.Parent();
  }

  if (!active_.count(node)) {
    return {};
  }

  // 2) collect all leaves under that ancestor
  std::vector<Loc> leaves;
  std::function<void(const Loc&)> recurse = [&](auto const& n) {
    if (IsLeaf(n)) {
      leaves.push_back(n);
    } else {
      for (auto& c : GetChildLocs(n)) {
        recurse(c);
      }
    }
  };

  recurse(node);
  return leaves;
}

}  // namespace topo
