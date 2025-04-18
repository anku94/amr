#pragma once

namespace topo {
//
// NeighborBlock: represents a single neighbor relation
//
struct NeighborBlock {
  int block_id;   // our block id
  int peer_rank;  // rank hosting nbr block
  int buf_id;     // ??
  int msg_sz;     // ??j
  int tag;        // tag for communication
};

//
// CommTags: tags for different types of communication
//
enum class CommTags {
    kFaceTag = 1,
    kEdgeTag = 2,
    kVertexTag = 3,
};

}