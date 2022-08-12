//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "block.h"
#include "common.h"

class Topology {
 public:
  static Status GenerateMesh(const DriverOpts& opts, Mesh& mesh) {
    switch (opts.topology) {
      case NeighborTopology::Ring:
        return GenerateMeshRing(opts, mesh);
      case NeighborTopology::AllToAll:
        return GenerateMeshAllToAll(opts, mesh);
      default:
        return Status::Error;
    }
  }

 private:
  static Status GenerateMeshRing(const DriverOpts& opts, Mesh& mesh) {
    for (size_t i = 0; i < opts.blocks_per_rank; i++) {
      int ring_delta = i * Globals::nranks;
      int bid_rel = Globals::my_rank;
      int nbr_left =
          ((bid_rel - 1) % Globals::nranks + Globals::nranks) % Globals::nranks;
      int nbr_right = (bid_rel + 1) % Globals::nranks;

      auto mb = std::make_shared<MeshBlock>(ring_delta + bid_rel);
      mb->AddNeighbor(nbr_left + ring_delta, nbr_left);
      mb->AddNeighbor(nbr_right + ring_delta, nbr_right);

      mesh.AddBlock(mb);
    }

    return Status::OK;
  }

  static Status GenerateMeshAllToAll(const DriverOpts& opts, Mesh& mesh) {
    if (!(opts.blocks_per_rank > Globals::nranks) and
        (opts.blocks_per_rank % Globals::nranks == 0)) {
      logf(LOG_ERRO, "Invalid arguments");
      ABORT("Invalid arguments");
    }

    int n = Globals::nranks;
    // blocks on rank i = n*i to n*i + (n - 1)

    for (size_t i = 0; i < n; i++) {
      int bid_i = n * Globals::my_rank + i;
      // neighboring rank, left
      int nrl = ((Globals::my_rank - i) % n + n) % n;
      // neighboring rank, right
      int nrr = (Globals::my_rank + i) % n;
      int nrl_bid = n * nrl + i;
      int nrr_bid = n * nrr + i;

      int nreps = opts.blocks_per_rank / n;
      for (int rep = 0; rep < nreps; rep++) {
        int off = n * n * rep;
        int bid_i_off = bid_i + off;
        int nrl_bid_off = nrl_bid + off;
        int nrr_bid_off = nrr_bid + off;
        logf(LOG_DBG2, "Block %d, Neighbors %d-%d", bid_i_off, nrl_bid_off,
             nrr_bid_off);

        auto mb = std::make_shared<MeshBlock>(bid_i_off);
        mb->AddNeighbor(nrl_bid_off, nrl);
        mb->AddNeighbor(nrr_bid_off, nrr);
        mesh.AddBlock(mb);
      }
    }

    return Status::OK;
  }
};