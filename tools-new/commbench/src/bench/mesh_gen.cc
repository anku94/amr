#include "mesh_gen.h"

#include "amr/block.h"
#include "tools-common/globals.h"

namespace {
// using namespace topo;
using namespace topo;

static void GatherNeighborCounts(int count_local) {
  std::vector<int> counts(Globals::nranks);

  int rv = MPI_Gather(&count_local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                      MPI_COMM_WORLD);
  if (rv != MPI_SUCCESS) {
    MLOG(MLOG_ERRO, "MPI_Allgather failed");
    ABORT("MPI_Allgather failed");
  }

  if (Globals::my_rank != 0) {
    return;
  }

  const char *fname = "neighbor_counts.txt";
  std::string fpath = std::string("/tmp/") + fname;
  // std::string(amr::Globals::driver_opts.job_dir) + "/" + fname;
  FILE *f = fopen(fpath.c_str(), "w");
  if (f == nullptr) {
    MLOG(MLOG_ERRO, "Failed to open file: %s", fpath.c_str());
    return;
  }

  for (int i = 0; i < Globals::nranks - 1; i++) {
    fprintf(f, "%d,", counts[i]);
  }

  fprintf(f, "%d\n", counts[Globals::nranks - 1]);
  fclose(f);
}
}  // namespace

namespace topo {
std::unique_ptr<MeshGenerator> MeshGenerator::Create(const DriverOpts &opts) {
  MeshGenMethod t = opts.meshgen_method;

  switch (t) {
    case MeshGenMethod::Ring:
      return std::make_unique<RingMeshGenerator>(opts);
      break;
    case MeshGenMethod::AllToAll:
      return std::make_unique<AllToAllMeshGenerator>(opts);
      break;
    case MeshGenMethod::Dynamic:
      ABORT("Not implemented");
      break;
    case MeshGenMethod::FromSingleTSTrace:
      return std::make_unique<SingleTimestepTraceMeshGenerator>(opts);
      break;
    case MeshGenMethod::FromMultiTSTrace:
      return std::make_unique<MultiTimestepTraceMeshGenerator>(opts);
      break;
  }

  return nullptr;
}

Status RingMeshGenerator::GenerateMesh(CommMesh &mesh, int ts) {
  for (size_t i = 0; i < opts_.blocks_per_rank; i++) {
    int ring_delta = i * Globals::nranks;
    int bid_rel = Globals::my_rank;
    int nbr_left =
        ((bid_rel - 1) % Globals::nranks + Globals::nranks) % Globals::nranks;
    int nbr_right = (bid_rel + 1) % Globals::nranks;

    auto nbr_tag = static_cast<int>(CommTags::kFaceTag);

    auto mb = std::make_shared<topo::MeshBlock>(ring_delta + bid_rel);
    mb->AddNeighborSendRecv(nbr_left + ring_delta, nbr_left, opts_.size_per_msg,
                            nbr_tag);
    mb->AddNeighborSendRecv(nbr_right + ring_delta, nbr_right,
                            opts_.size_per_msg, nbr_tag);

    MeshGenerator::AddMeshBlock(mesh, mb);
  }

  return Status::OK;
}

Status AllToAllMeshGenerator::GenerateMesh(CommMesh &mesh, int ts) {
  if (opts_.blocks_per_rank <= Globals::nranks and
      (opts_.blocks_per_rank % Globals::nranks == 0)) {
    MLOG(MLOG_ERRO, "Invalid arguments");
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
      MLOG(MLOG_DBG2, "Block %d, Neighbors %d-%d", bid_i_off, nrl_bid_off,
           nrr_bid_off);

      // Assume all msgs are face msgs
      auto nbr_tag = static_cast<int>(CommTags::kFaceTag);

      auto mb = std::make_shared<MeshBlock>(bid_i_off);
      mb->AddNeighborSendRecv(nrl_bid_off, nrl, opts_.size_per_msg, nbr_tag);
      mb->AddNeighborSendRecv(nrr_bid_off, nrr, opts_.size_per_msg, nbr_tag);

      MeshGenerator::AddMeshBlock(mesh, mb);
    }
  }
  return Status::OK;
}

Status SingleTimestepTraceMeshGenerator::GenerateMesh(CommMesh &mesh, int ts) {
  Status s = Status::OK;

  MLOG(MLOG_INFO, "Generating Mesh: From Trace (ts: %d)", ts);

  s = reader_.Read(Globals::my_rank);

  auto msgs_snd = reader_.GetMsgsSent();
  auto msgs_rcv = reader_.GetMsgsRcvd();

  if (msgs_snd.size() != msgs_rcv.size()) {
    MLOG(MLOG_WARN, "[Rank %d] msg_send count is not the same as msg_rcv count",
         Globals::my_rank);

    // in the trace replay mode, this is not a bug.
    // as all ranks see the same trace, they create a consistent
    // communication graph
  }

  int mbidx = 0;
  std::vector<std::shared_ptr<MeshBlock>> mb_vec;
  auto mb = std::make_shared<MeshBlock>(0);

  int nbr_tag =
      static_cast<int>(CommTags::kFaceTag);  // Assume all msgs are face msgs
  int nbr_idx = 0;

  for (auto it : msgs_snd) {
    mb->AddNeighborSend(nbr_idx++, it.peer_rank, it.msg_sz, nbr_tag);
  }

  for (auto it : msgs_rcv) {
    mb->AddNeighborRecv(nbr_idx++, it.peer_rank, it.msg_sz, nbr_tag);
  }

  MeshGenerator::AddMeshBlock(mesh, mb);

  MLOGIFR0(MLOG_INFO,
           "[GenerateMeshFromTrace] Rank: %d, Neighbors: %d\n"
           "(full log in neighbor_counts.txt)",
           Globals::my_rank, nbr_idx);

  GatherNeighborCounts(nbr_idx);

  return s;
}

Status MultiTimestepTraceMeshGenerator::GenerateMesh(CommMesh &mesh, int ts) {
  Status s = Status::OK;

  MLOG(MLOG_INFO, "Generating Mesh: From Trace (ts: %d)", ts);

  s = reader_.Read(Globals::my_rank);

  auto msgs_snd = reader_.GetMsgsSent(ts);
  auto msgs_rcv = reader_.GetMsgsRcvd(ts);

  if (msgs_snd.size() != msgs_rcv.size()) {
    MLOG(MLOG_WARN, "[Rank %d] msg_send count is not the same as msg_rcv count",
         Globals::my_rank);

    // in the trace replay mode, this is not a bug.
    // as all ranks see the same trace, they create a consistent
    // communication graph
  }

  int mbidx = 0;
  std::vector<std::shared_ptr<MeshBlock>> mb_vec;
  auto mb = std::make_shared<MeshBlock>(0);

  int nbr_idx = 0;
  // Assume all msgs are face msgs
  int nbr_tag = static_cast<int>(CommTags::kFaceTag);

  for (auto it : msgs_snd) {
    mb->AddNeighborSend(nbr_idx++, it.peer_rank, it.msg_sz, nbr_tag);
  }

  for (auto it : msgs_rcv) {
    mb->AddNeighborRecv(nbr_idx++, it.peer_rank, it.msg_sz, nbr_tag);
  }

  MeshGenerator::AddMeshBlock(mesh, mb);

  MLOGIFR0(MLOG_INFO,
           "[GenerateMeshFromTrace] Rank: %d, Neighbors: %d\n"
           "(full log in neighbor_counts.txt)",
           Globals::my_rank, nbr_idx);

  GatherNeighborCounts(nbr_idx);

  return s;
}
}  // namespace topo