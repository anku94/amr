#pragma once

#include <memory>

namespace topo {
class MeshBlock;  // fwd decl
}

using MeshBlockRef = std::shared_ptr<topo::MeshBlock>;
