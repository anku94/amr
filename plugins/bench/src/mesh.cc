#include "mesh.h"

namespace amr {
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
}  // namespace amr