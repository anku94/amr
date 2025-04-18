#pragma once

#include <cinttypes>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace topo {
// OrderedID: single block's SFC ID
using OrderedBlockVec = std::vector<int>;

//
// OrderedMeshNode: a single block and its neighbors
// in SFC block ID format
//
struct OrderedMeshNode {
  OrderedBlockVec face, edge, vertex;

  // PackOne: pack a single vector into a stream for MPI bcast
  static void PackOne(const OrderedBlockVec& vec, std::vector<int>& pack_out) {
    pack_out.push_back(vec.size());
    pack_out.insert(pack_out.end(), vec.begin(), vec.end());
  }

  // Pack: pack all ovecs into streams
  void Pack(std::vector<int>& pack_out) const {
    PackOne(face, pack_out);
    PackOne(edge, pack_out);
    PackOne(vertex, pack_out);
  }

  // UnpackOne: unpack one ovec from stream
  static void UnpackOne(OrderedBlockVec& vec, std::vector<int> const& pack_in,
                        int& cur) {
    int n = pack_in[cur++];
    vec.resize(n);
    std::copy(pack_in.begin() + cur, pack_in.begin() + cur + n, vec.begin());
    cur += n;
  }

  // Unpack: unpack all ovecs from stream
  void Unpack(std::vector<int> const& pack_in, int& cur) {
    UnpackOne(face, pack_in, cur);
    UnpackOne(edge, pack_in, cur);
    UnpackOne(vertex, pack_in, cur);
  }
};

//
// OrderedMesh: mesh structure in SFC block IDs
//
struct OrderedMesh {
  int nblocks;                          // total leaf blocks
  std::vector<OrderedMeshNode> nbrmap;  // [blockid] -> nbrs[]

  // Pack: pack all vectors into a stream for MPI bcast
  void Pack(std::vector<int>& pack_out) const {
    pack_out.clear();
    pack_out.push_back(nblocks);
    for (const auto& node : nbrmap) {
      node.Pack(pack_out);
    }
  }

  // Unpack: unpack all vectors from a stream
  void Unpack(std::vector<int> const& pack_in) {
    int cur = 0;
    nblocks = pack_in[cur++];
    nbrmap.resize(nblocks);
    for (auto& node : nbrmap) {
      node.Unpack(pack_in, cur);
    }
  }
};

// Vec3: 3D vector utility class
template <typename T>
struct Vec3 {
  T x, y, z;

  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  Vec3() : x(0), y(0), z(0) {}

  Vec3 operator+(const Vec3& other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
  }

  Vec3 operator-(const Vec3& other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
  }

  Vec3 operator>>(int shift) const {
    return Vec3(x >> shift, y >> shift, z >> shift);
  }

  Vec3 operator<<(int shift) const {
    return Vec3(x << shift, y << shift, z << shift);
  }

  bool operator==(const Vec3& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  // AnyLT: any dim less than that of other
  bool AnyLT(T val) const { return x < val || y < val || z < val; }
  // AnyGT: any dim greater than that of other
  bool AnyGT(T val) const { return x > val || y > val || z > val; }
  // AnyLTE: any dim less than or equal to that of other
  bool AnyLTE(T val) const { return x <= val && y <= val && z <= val; }
  // AnyGTE: any dim greater than or equal to that of other
  bool AnyGTE(T val) const { return x >= val && y >= val && z >= val; }

  // ToString: convert to string
  std::string ToString() const {
    std::stringstream ss;
    ss << "Vec3(" << x << ", " << y << ", " << z << ")";
    return ss.str();
  }
};

using Vec3i = Vec3<int>;
using Vec3ll = Vec3<int64_t>;

// Logical location in the AMR hierarchy
// locv of a child is (locv_parent << 1) + offset
// offset: [0, 8). each offset bit is added to a locv dim
struct Loc {
  int level = -1;  // refinement level
  Vec3ll locv;     // Loc vec

  Loc() : level(-1), locv{-1, -1, -1} {}
  Loc(int level, Vec3ll loc) : level(level), locv(loc) {}

  std::string ToString() const {
    char buf[128];
    snprintf(buf, sizeof(buf), "Loc[L%d]: %" PRId64 ", %" PRId64 ", %" PRId64,
             level, locv.x, locv.y, locv.z);
    return std::string(buf);
  }

  bool operator==(const Loc& o) const {
    return level == o.level && locv == o.locv;
  }

  bool operator!=(const Loc& o) const { return !(*this == o); }

  bool IsValid() const { return level >= 0; }

  Loc Parent() const { return (level > 0) ? Loc{level - 1, locv >> 1} : Loc(); }
};

inline std::ostream& operator<<(std::ostream& os, const Loc& loc) {
  os << loc.ToString();
  return os;
}

// Hash for Loc in unordered containers
struct LocHash {
  size_t operator()(const Loc& loc) const noexcept {
    size_t h = std::hash<int>()(loc.level);
    h ^= std::hash<int64_t>()(loc.locv.x) << 1;
    h ^= std::hash<int64_t>()(loc.locv.y) << 2;
    h ^= std::hash<int64_t>()(loc.locv.z) << 3;
    return h;
  }
};
}  // namespace topo
