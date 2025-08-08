#pragma once

#include "amr/amr_types.h"
#include "amr/block.h"
#include "amr_lb.h"
#include "bench/comm_mesh.h"
#include "tools-common/globals.h"

namespace topo {
class PlacementUtils {
 public:
  // SetupCommMesh: setup comm mesh from ordered mesh after placement
  static int SetupCommMesh(CommMesh &comm_mesh, const OrderedMesh &omesh,
                           std::vector<int> const &ranklist,
                           topo::Vec3i const &msgsz) {
    auto blocks_ =
        CreateBlocksFromOmesh(omesh, ranklist, Globals::my_rank, msgsz);
    for (const auto &block : blocks_) {
      Status s = comm_mesh.AddBlock(block);
      MLOGIF(s != Status::OK, MLOG_ERRO, "Block add failed");
    }

    return 0;
  }

 private:
  // CreateBlocksFromOmesh: create blocks from ordered mesh
  // and set their neighbors up
  static std::vector<MeshBlockRef> CreateBlocksFromOmesh(
      const OrderedMesh &omesh, const std::vector<int> &ranklist, int my_rank,
      topo::Vec3i const &msgsz) {
    std::vector<MeshBlockRef> blocks;

    auto f_msgsz = msgsz.x;
    auto e_msgsz = msgsz.y;
    auto v_msgsz = msgsz.z;

    auto rank_bids = GetBlockidsByRank(ranklist, my_rank);

    for (int bid : rank_bids) {
      auto block = std::make_shared<MeshBlock>(bid);

      MLOGIFR0(MLOG_DBG0, "Setting up nbrs for block %d", bid);
      auto &fvec = omesh.nbrmap[bid].face;
      auto ftag = static_cast<int>(CommTags::kFaceTag);
      AddNeighborVec(block, fvec, ranklist, f_msgsz, ftag);

      auto &evec = omesh.nbrmap[bid].edge;
      auto etag = static_cast<int>(CommTags::kEdgeTag);
      AddNeighborVec(block, evec, ranklist, e_msgsz, etag);

      auto &vvec = omesh.nbrmap[bid].vertex;
      auto vtag = static_cast<int>(CommTags::kVertexTag);
      AddNeighborVec(block, vvec, ranklist, v_msgsz, vtag);

      blocks.push_back(block);
    }

    return blocks;
  }

  // AddNeighbor: add a neighbor vec (face/edge/vtx) to the block
  static void AddNeighborVec(MeshBlockRef &block, std::vector<int> bids,
                             std::vector<int> ranklist, int msg_sz, int tag) {
    MLOGIFR0(MLOG_DBG1, "- Adding %zu nbrs", bids.size());

    for (int bid : bids) {
      int peer_rank = ranklist[bid];
      block->AddNeighborSendRecv(bid, peer_rank, msg_sz, tag);
    }
  }

  // GetBlockidsByRank: get blockids whose rank matches ours from ranklist
  static std::vector<int> GetBlockidsByRank(const std::vector<int> &ranklist,
                                            int my_rank) {
    std::vector<int> rank_bids;

    int nblocks = ranklist.size();
    for (int bid = 0; bid < nblocks; ++bid) {
      if (ranklist[bid] == my_rank) {
        rank_bids.push_back(bid);
      }
    }

    MLOG(MLOG_DBG0, "Found %zu blocks for rank %d", rank_bids.size(), my_rank);

    return rank_bids;
  }
};
}  // namespace topo
