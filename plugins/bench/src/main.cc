
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "mesh.h"
#include "print_utils.h"
#include "logging.h"

using namespace amr;

int main() {
  // Create a small root mesh of 2x2x2 with max refinement level 2
  Mesh mesh(2, 2, 2, 3);

  // Print initial mesh hierarchy
  LOG(LOG_INFO, "--- Initial mesh hierarchy: ---\n");
  Loc root{mesh.GetRootLevel(), {0, 0, 0}};
  PrintUtils::PrintHierarchy(mesh, root);

  // Refine root block
  LOG(LOG_INFO, "\n--- Refining root block: ---\n");
  mesh.Refine(root);

  // Print hierarchy after refinement
  LOG(LOG_INFO, "\n--- Mesh hierarchy after one refinement: ---\n");
  PrintUtils::PrintHierarchy(mesh, {0, {0, 0, 0}});

  // Refine one of the new children
  // Loc child1{root.level + 1, root.locv << 1};
  // LOG(LOG_INFO, "\nRefining child block at %s...\n", child1.ToString().c_str());
  // mesh.Refine(child1);

  // // Print hierarchy after second refinement
  // LOG("\n--- Mesh hierarchy after second refinement: ---\n");
  // PrintUtils::PrintHierarchy(mesh, root);

  // Print ordered mesh
  Mesh::OrderedMesh om = mesh.GetOrderedMesh();

  LOG(LOG_INFO, "\n--- Ordered mesh: ---\n");
  PrintUtils::PrintOmesh(om);

  LOG(LOG_INFO, "--- Validating ordered mesh: ---\n");
  PrintUtils::ValidateOmesh(om);

  return 0;
}