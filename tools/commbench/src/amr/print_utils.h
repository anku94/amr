#pragma once

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "mesh.h"
#include "tools-common/logging.h"
namespace topo {
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

    if (indent == 0) {
      MLOG(MLOG_INFO, "-- Mesh hierarchy (total %d blocks, %d leaves) --\n",
           mesh.GetTotalBlockCount(), mesh.GetLeafBlockCount());
    }
    auto children = Mesh::GetChildLocs(loc);

    // Gather direct leaf children codes
    std::vector<std::string> leafCodes;
    leafCodes.reserve(8);
    for (auto& c : children) {
      if (mesh.IsLeaf(c)) leafCodes.push_back(Code(c));
    }

    auto leafcodes_str = SerializeVec(leafCodes);

    // Log this node and its leaves
    MLOG(MLOG_INFO, "%*sNode %s: %s (Leaves: %s)", indent * 2, "",
         Code(loc).c_str(), loc.ToString().c_str(), leafcodes_str.c_str());

    // Recurse into non-leaf children
    for (auto& c : children) {
      if (!mesh.IsLeaf(c) && mesh.IsActive(c)) {
        PrintHierarchy(mesh, c, indent + 1);
      }
    }
  }

  template <typename T>
  static std::string SerializeVec(std::vector<T>& vec, std::string sep = ", ", int max_elems = -1) {
    // sort vec first
    // std::sort(vec.begin(), vec.end());
    std::ostringstream oss;

    // serialize vec
    bool first_elem = true;
    int count = 0;

    for (auto& elem : vec) {
      if (!first_elem) oss << sep;
      oss << elem;
      first_elem = false;
      count++;

      if (max_elems != -1 && count >= max_elems) {
        oss << "...";
        break;
      }
    }

    return oss.str();
  }

  static void PrintOmeshBlock(OrderedMesh& om, int bid) {
    auto& block = om.nbrmap[bid];
    int facecnt = block.face.size();
    int edgecnt = block.edge.size();
    int vercnt = block.vertex.size();
    int totcnt = facecnt + edgecnt + vercnt;

    MLOG(MLOG_DBG0, "Block %2d: %3d nbrs (f/e/v: %2d/%2d/%2d)\n", bid, totcnt,
         facecnt, edgecnt, vercnt);

    MLOG(MLOG_DBG1, "  -- [face] %s, [edge] %s, [vert] %s",
         SerializeVec(block.face).c_str(), SerializeVec(block.edge).c_str(),
         SerializeVec(block.vertex).c_str());
  }

  static void PrintOmesh(OrderedMesh& om) {
    MLOG(MLOG_INFO, "--- Ordered mesh with %d blocks ---\n", om.nblocks);
    for (int i = 0; i < om.nblocks; ++i) {
      PrintOmeshBlock(om, i);
    }
  }

  static void AssertValInVec(const OrderedBlockVec& vec, int val, int vec_id,
                             const char* vec_name) {
    auto it = std::find(vec.begin(), vec.end(), val);
    if (it == vec.end()) {
      MLOG(MLOG_ERRO, "!ERROR! Block %d not found in vec::%s of block %d\n",
           val, vec_name, vec_id);
      exit(1);
    }
  }

  // for each block i, if it has a nbr j, j should have i
  static void ValidateOmesh(const OrderedMesh& om) {
    for (int i = 0; i < om.nblocks; ++i) {
      MLOG(MLOG_INFO, "Validating block %d...", i);

      MLOG(MLOG_INFO, " checking faces ...");
      for (int j : om.nbrmap[i].face) {
        AssertValInVec(om.nbrmap[j].face, i, j, "face");
      }

      MLOG(MLOG_INFO, " checking edges ...");
      for (int j : om.nbrmap[i].edge) {
        AssertValInVec(om.nbrmap[j].edge, i, j, "edge");
      }

      MLOG(MLOG_INFO, " checking vertices ...");
      for (int j : om.nbrmap[i].vertex) {
        AssertValInVec(om.nbrmap[j].vertex, i, j, "vertex");
      }

      MLOG(MLOG_INFO, " done\n");
    }

    MLOG(MLOG_INFO, "Validation passed\n");
  }
};
}  // namespace topo
