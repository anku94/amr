#pragma once

#include <memory>

namespace topo {
namespace amr {
class MeshBlock; // fwd decl
}

using MeshBlockRef = std::shared_ptr<amr::MeshBlock>;
}  // namespace topo