#pragma once

#include <cstdint>
#include <functional>

namespace amr {
template <typename T>
struct Vec3 {
  T x, y, z;

  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  Vec3() : x(0), y(0), z(0) {}

  Vec3 operator+(const Vec3& other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
  }
};

using Vec3i = Vec3<int>;

// Logical location in the AMR hierarchy
struct Loc {
  int level = -1;
  int64_t lx = -1, ly = -1, lz = -1;

  bool operator==(const Loc& o) const {
    return level == o.level && lx == o.lx && ly == o.ly && lz == o.lz;
  }

  bool operator!=(const Loc& o) const { return !(*this == o); }

  bool IsValid() const { return level >= 0; }

  Loc Parent() const {
    return (level > 0) ? Loc{level - 1, lx >> 1, ly >> 1, lz >> 1} : Loc();
  }
};

// Hash for Loc in unordered containers
struct LocHash {
  size_t operator()(const Loc& loc) const noexcept {
    size_t h = std::hash<int>()(loc.level);
    h ^= std::hash<int64_t>()(loc.lx) << 1;
    h ^= std::hash<int64_t>()(loc.ly) << 2;
    h ^= std::hash<int64_t>()(loc.lz) << 3;
    return h;
  }
};
}  // namespace amr