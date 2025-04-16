#include "mesh.h"

#include <cstdio>
#include <vector>
#include <string>

#define LOG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)

#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include "mesh.h"

#define LOG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)

using namespace amr;

struct Utils {
  // Short code for a location: e.g. "012"
  static std::string Code(const Loc& loc) {
    std::ostringstream oss;
    oss << loc.lx << loc.ly << loc.lz;
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
    LOG("%*sNode %s L%d (%lld,%lld,%lld): Leaves:",
        indent*2, "",
        Code(loc).c_str(), loc.level, loc.lx, loc.ly, loc.lz);
    for (auto& code : leafCodes) LOG(" %s", code.c_str());
    LOG("\n");

    // Recurse into non-leaf children
    for (auto& c : children) {
      if (!mesh.IsLeaf(c)) {
        PrintHierarchy(mesh, c, indent + 1);
      }
    }
  }
};

int main() {
  // Create a small root mesh of 2x2x2 with max refinement level 2
  Mesh mesh(2, 2, 2, 3);

  // Print initial mesh hierarchy
  LOG("Initial mesh hierarchy:\n");
  Loc root{mesh.GetRootLevel(), 0, 0, 0};
  Utils::PrintHierarchy(mesh, root);

  // Refine root block
  LOG("\nRefining root block...\n");
  mesh.Refine(root);

  // Print hierarchy after refinement
  LOG("Mesh hierarchy after one refinement:\n");
  Utils::PrintHierarchy(mesh, root);

  // Refine one of the new children
  Loc child1{root.level + 1, (root.lx << 1), (root.ly << 1), (root.lz << 1)};
  LOG("\nRefining child block at (%lld, %lld, %lld) level %d...\n", child1.lx,
      child1.ly, child1.lz, child1.level);
  mesh.Refine(child1);

  // Print hierarchy after second refinement
  LOG("Mesh hierarchy after second refinement:\n");
  Utils::PrintHierarchy(mesh, root);

  return 0;
}