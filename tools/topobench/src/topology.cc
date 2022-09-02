//
// Created by Ankush J on 4/11/22.
//

#include "topology.h"

Status Topology::GenerateMesh(const DriverOpts& opts, Mesh& mesh, int ts) {
  // TODO: clear old mesh first
  switch (opts.topology) {
    case NeighborTopology::Ring:
      return GenerateMeshRing(mesh, ts);
      break;
    case NeighborTopology::AllToAll:
      return GenerateMeshAllToAll(mesh, ts);
      break;
    case NeighborTopology::Dynamic:
      return GenerateMeshDynamic(mesh, ts);
      break;
    case NeighborTopology::FromTrace:
      return GenerateMeshFromTrace(mesh, ts);
      break;
  }

  return Status::Error;
}

Status Topology::GenerateMeshRing(Mesh& mesh, int ts) const {
  for (size_t i = 0; i < opts_.blocks_per_rank; i++) {
    int ring_delta = i * Globals::nranks;
    int bid_rel = Globals::my_rank;
    int nbr_left =
        ((bid_rel - 1) % Globals::nranks + Globals::nranks) % Globals::nranks;
    int nbr_right = (bid_rel + 1) % Globals::nranks;

    auto mb = std::make_shared<MeshBlock>(ring_delta + bid_rel);
    mb->AddNeighborSendRecv(nbr_left + ring_delta, nbr_left,
                            opts_.size_per_msg);
    mb->AddNeighborSendRecv(nbr_right + ring_delta, nbr_right,
                            opts_.size_per_msg);

    mesh.AddBlock(mb);
  }

  return Status::OK;
}

Status Topology::GenerateMeshAllToAll(Mesh& mesh, int ts) const {
  if (opts_.blocks_per_rank <= Globals::nranks and
      (opts_.blocks_per_rank % Globals::nranks == 0)) {
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

    int nreps = opts_.blocks_per_rank / n;
    for (int rep = 0; rep < nreps; rep++) {
      int off = n * n * rep;
      int bid_i_off = bid_i + off;
      int nrl_bid_off = nrl_bid + off;
      int nrr_bid_off = nrr_bid + off;
      logf(LOG_DBG2, "Block %d, Neighbors %d-%d", bid_i_off, nrl_bid_off,
           nrr_bid_off);

      auto mb = std::make_shared<MeshBlock>(bid_i_off);
      mb->AddNeighborSendRecv(nrl_bid_off, nrl, opts_.size_per_msg);
      mb->AddNeighborSendRecv(nrr_bid_off, nrr, opts_.size_per_msg);
      mesh.AddBlock(mb);
    }
  }
  return Status::OK;
}

Status Topology::GenerateMeshFromTrace(Mesh& mesh, int ts) {
  Status s = Status::OK;

  logf(LOG_INFO, "Generating Mesh: From Trace");

  s = reader_.Read();

  std::vector<RankSizePair> msgs_snd = reader_.GetMsgsSent(ts);
  std::vector<RankSizePair> msgs_rcv = reader_.GetMsgsRcvd(ts);

  if (msgs_snd.size() != msgs_rcv.size()) {
    logf(LOG_ERRO, "msg_send count is not the same as msg_rcv count");
    return Status::Error;
  }

  int mbidx = 0;
  std::vector<std::shared_ptr<MeshBlock>> mb_vec;

  for (auto it : msgs_snd) {
    int peer = it.first;
    int msgsz = it.second;

    auto mb = std::make_shared<MeshBlock>(mbidx);
    mb->AddNeighborSend(mbidx, peer, msgsz);
    mb_vec.push_back(mb);
  }

  mbidx = 0;
  for (auto it : msgs_rcv) {
    int peer = it.first;
    int msgsz = it.second;

    mb_vec[mbidx]->AddNeighborRecv(mbidx, peer, msgsz);
    mesh.AddBlock(mb_vec[mbidx]);
  }

  return s;
}
