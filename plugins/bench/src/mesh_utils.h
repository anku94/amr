#pragma once

#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace amr {
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

  bool AnyLT(T val) const { return x < val || y < val || z < val; }
  bool AnyGT(T val) const { return x > val || y > val || z > val; }
  bool AnyLTE(T val) const { return x <= val && y <= val && z <= val; }
  bool AnyGTE(T val) const { return x >= val && y >= val && z >= val; }
};

using Vec3i = Vec3<int>;
using Vec3ll = Vec3<int64_t>;

// Logical location in the AMR hierarchy
struct Loc {
  int level = -1;
  Vec3ll locv;

  Loc() : level(-1), locv{-1, -1, -1} {}
  Loc(int level, Vec3ll loc) : level(level), locv(loc) {}

  std::string ToString() const {
    char buf[128];
    snprintf(buf, sizeof(buf), "Loc[L%d]: %lld, %lld, %lld", level, locv.x,
             locv.y, locv.z);
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
}  // namespace amr