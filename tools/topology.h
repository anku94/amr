//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "block.h"
#include "common.h"

class Topology {
 public:
  Status GenerateMesh(const DriverOpts &opts) {
    switch (opts.topology) {
      case NeighborTopology::Ring:
        return GenerateMeshRing(opts);
      case NeighborTopology::AllToAll:
        return GenerateMeshAllToAll(opts);
      default:
        return Status::Error;
    }
  }

 private:
  Status GenerateMeshRing(const DriverOpts &opts) {
    Mesh mesh;

    for (size_t i = 0; i < opts.blocks_per_rank; i++) {
      int ring_delta = i * Globals::nranks;
      int bid_rel = Globals::my_rank;
      int nbr_left = (bid_rel - 1) % Globals::nranks;
      int nbr_right = (bid_rel + 1) % Globals::nranks;

      MeshBlock mb(ring_delta + bid_rel);
      mb.AddNeighbor(nbr_left + ring_delta, nbr_left);
      mb.AddNeighbor(nbr_right + ring_delta, nbr_right);

      mesh.AddBlock(mb);
    }

    return Status::OK;
  }

  Status GenerateMeshAllToAll(const DriverOpts &opts) {
    return Status::OK;
  }
};