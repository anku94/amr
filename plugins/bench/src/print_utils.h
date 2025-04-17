#pragma once

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "mesh.h"
#include "logging.h"

namespace amr {
struct PrintUtils {
  // Short code for a location: e.g. "012"
  static std::string Code(const Loc& loc) {
    std::ostringstream oss;
    oss << loc.locv.x << loc.locv.y << loc.locv.z;
    return oss.str();
  }

  // Print internal nodes and their direct leaf children inline
  static void PrintHierarchy(const Mesh& mesh, const Loc& loc, int indent = 0) {
    if (mesh.IsLeaf(loc)) return;
    auto children = Mesh::GetChildLocs(loc);

    // Gather direct leaf children codes
    std::vector<std::string> leafCodes;
    leafCodes.reserve(8);
    for (auto& c : children) {
      if (mesh.IsLeaf(c)) leafCodes.push_back(Code(c));
    }

    // Log this node and its leaves
    LOG(LOG_INFO, "%*sNode %s L%d (%lld,%lld,%lld): Leaves:", indent * 2, "",
        Code(loc).c_str(), loc.level, loc.locv.x, loc.locv.y, loc.locv.z);
    for (auto& code : leafCodes) LOG(LOG_INFO, " %s", code.c_str());
    LOG(LOG_INFO, "\n");

    // Recurse into non-leaf children
    for (auto& c : children) {
      if (!mesh.IsLeaf(c)) {
        PrintHierarchy(mesh, c, indent + 1);
      }
    }
  }

  template <typename T>
  static std::string SerializeVec(std::vector<T>& vec) {
    // sort vec first
    // std::sort(vec.begin(), vec.end());
    std::ostringstream oss;

    // serialize vec
    bool first_elem = true;
    for (auto& elem : vec) {
      if (!first_elem) oss << ", ";
      oss << elem;
      first_elem = false;
    }

    return oss.str();
  }

  static void PrintOmeshBlock(Mesh::OrderedMesh& om, int bid) {
    auto& block = om.nbrmap[bid];
    int facecnt = block.face.size();
    int edgecnt = block.edge.size();
    int vercnt = block.vertex.size();
    int totcnt = facecnt + edgecnt + vercnt;

    LOG(LOG_INFO, "Block %2d: %3d nbrs (f/e/v: %2d/%2d/%2d)\n", bid, totcnt,
        facecnt, edgecnt, vercnt);

    LOG(LOG_INFO, "  - face nbrs: %s\n", SerializeVec(block.face).c_str());
    LOG(LOG_INFO, "  - edge nbrs: %s\n", SerializeVec(block.edge).c_str());
    LOG(LOG_INFO, "  - vertex nbrs: %s\n", SerializeVec(block.vertex).c_str());
  }

  static void PrintOmesh(Mesh::OrderedMesh& om) {
    LOG(LOG_INFO, "Ordered mesh with %d blocks:\n", om.nblocks);
    for (int i = 0; i < om.nblocks; ++i) {
      PrintOmeshBlock(om, i);
    }
  }

  static void AssertValInVec(const Mesh::OrderedBlockVec& vec, int val,
                             int vec_id, const char* vec_name) {
    auto it = std::find(vec.begin(), vec.end(), val);
    if (it == vec.end()) {
      LOG(LOG_ERROR, "!ERROR! Block %d not found in vec::%s of block %d\n", val,
          vec_name, vec_id);
      exit(1);
    }
  }

  // for each block i, if it has a nbr j, j should have i
  static void ValidateOmesh(const Mesh::OrderedMesh& om) {
    for (int i = 0; i < om.nblocks; ++i) {
      LOG(LOG_INFO, "Validating block %d...", i);

      LOG(LOG_INFO, " checking faces ...");
      for (int j : om.nbrmap[i].face) {
        AssertValInVec(om.nbrmap[j].face, i, j, "face");
      }

      LOG(LOG_INFO, " checking edges ...");
      for (int j : om.nbrmap[i].edge) {
        AssertValInVec(om.nbrmap[j].edge, i, j, "edge");
      }

      LOG(LOG_INFO, " checking vertices ...");
      for (int j : om.nbrmap[i].vertex) {
        AssertValInVec(om.nbrmap[j].vertex, i, j, "vertex");
      }

      LOG(LOG_INFO, " done\n");
    }

    LOG(LOG_INFO, "Validation passed\n");
  }
};
}  // namespace amr
