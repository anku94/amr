#include "p2p.h"

#include "common.h"
#include "logging.h"
#include "print_utils.h"

#include <memory>
#include <mpi.h>
#include <vector>

#define SAFE_MPI(call, ret_val)                          \
  do {                                                   \
    int rv = call;                                       \
    if (rv != MPI_SUCCESS) {                             \
      Error(__LOG_ARGS__, "MPI call failed: %s", #call); \
      return ret_val;                                    \
    }                                                    \
  } while (0)

namespace amr {
int ComputeRanksPerNode(int nranks) {
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;

  SAFE_MPI(PMPI_Get_processor_name(processor_name, &name_len), -1);

  std::vector<char> all_names(nranks * MPI_MAX_PROCESSOR_NAME);
  SAFE_MPI(PMPI_Allgather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                          all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                          MPI_COMM_WORLD),
           -1);

  std::unordered_map<std::string, int> node_counts;
  for (int i = 0; i < nranks; ++i) {
    std::string name(&all_names[i * MPI_MAX_PROCESSOR_NAME]);
    node_counts[name]++;
  }

  std::unordered_map<int, int> count_of_counts;
  for (const auto& kv : node_counts) {
    count_of_counts[kv.second]++;
  }

  int count_mostcommon = -1;
  int countfreq_mostcommon = 0;
  for (const auto& kv : count_of_counts) {
    if (kv.second > countfreq_mostcommon) {
      count_mostcommon = kv.first;
      countfreq_mostcommon = kv.second;
    }
  }

  return count_mostcommon;
}

MatrixAnalysis AnalyzeMatrix(uint64_t* matrix, int nranks, int npernode) {
  uint64_t sum_local = 0;
  uint64_t sum_global = 0;

  for (int i = 0; i < nranks; i++) {
    for (int j = 0; j < nranks; j++) {
      auto val = matrix[i * nranks + j];

      // Assuming sequential allocation of ranks to nodes
      int inode = i / npernode;
      int jnode = j / npernode;

      sum_local += (inode == jnode) ? val : 0;
      sum_global += val;
    }
  }

  return {sum_local, sum_global};
}

void MPIMemDeleter(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  int rv = MPI_Free_mem(ptr);
  if (rv != MPI_SUCCESS) {
    Error(__LOG_ARGS__, "MPI_Free_mem failed");
  }
}

int GetIdx(int my_rank, int other_rank, int nranks, bool is_send) {
  int src_rank = -1, dest_rank = -1;
  if (is_send) {
    src_rank = my_rank;
    dest_rank = other_rank;
  } else {
    src_rank = other_rank;
    dest_rank = my_rank;
  }

  int idx = src_rank * nranks + dest_rank;
  return idx;
}

std::vector<uint64_t> MapToMatrix(const std::unordered_map<int, uint64_t>& m,
                                  int my_rank, int nranks, bool is_send) {
  std::vector<uint64_t> matrix(nranks * nranks, 0);
  for (const auto& kv : m) {
    int idx = GetIdx(my_rank, kv.first, nranks, is_send);
    matrix[idx] = kv.second;
  }

  return matrix;
}

MatrixAnalysis P2PCommCollector::CollectMatrixWithReduce(
    const std::unordered_map<int, uint64_t>& m, bool is_send) {
  std::vector<uint64_t> matrix_global(nranks_ * nranks_, 0);
  auto matrix_local = MapToMatrix(m, my_rank_, nranks_, is_send);

  MatrixAnalysis analysis{0, 0};

  SAFE_MPI(PMPI_Reduce(matrix_local.data(), matrix_global.data(),
                       matrix_local.size(), MPI_UINT64_T, MPI_SUM, 0,
                       MPI_COMM_WORLD),
           analysis);

  analysis = AnalyzeMatrix(matrix_global.data(), nranks_, npernode_);

  if (my_rank_ == 0) {
    Verbose(__LOG_ARGS__, 1, "Global Comm Matrix: \n%s\n",
            MetricPrintUtils::MatrixToStr(matrix_global, nranks_).c_str());
  }

  return analysis;
}

MatrixAnalysis P2PCommCollector::CollectMatrixWithPuts(
    const std::unordered_map<int, uint64_t>& map, bool is_send) {
  int rv = 0;
  MatrixAnalysis analysis{0, 0};
  uint64_t* matrix_global = nullptr;

  MPI_Win win;
  size_t matsz = nranks_ * nranks_ * sizeof(uint64_t);

  if (my_rank_ == 0) {
    PMPI_Alloc_mem(matsz, MPI_INFO_NULL, &matrix_global);
    memset(matrix_global, 0, matsz);
  }

  // Ensures that matrix_global is freed when function exits
  std::unique_ptr<uint64_t, decltype(&MPIMemDeleter)> matrix_global_ptr(
      matrix_global, MPIMemDeleter);

  SAFE_MPI(
      PMPI_Win_create(matrix_global, my_rank_ == 0 ? matsz : 0,
                      sizeof(uint64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win),
      analysis);

  SAFE_MPI(PMPI_Win_fence(0, win), analysis);

  for (const auto& kv : map) {
    int idx = GetIdx(my_rank_, kv.first, nranks_, is_send);
    auto val = kv.second;

    SAFE_MPI(PMPI_Put(&val, 1, MPI_UINT64_T, 0, idx, 1, MPI_UINT64_T, win),
             analysis);
  }

  SAFE_MPI(PMPI_Win_fence(0, win), analysis);
  SAFE_MPI(PMPI_Win_free(&win), analysis);

  if (my_rank_ == 0) {
    analysis = AnalyzeMatrix(matrix_global, nranks_, npernode_);
  }

  return analysis;
}

std::string P2PCommCollector::CollectAndAnalyze(int my_rank, int nranks) {
  my_rank_ = my_rank;
  nranks_ = nranks;
  npernode_ = ComputeRanksPerNode(nranks);

  if (my_rank == 0) {
    Info(__LOG_ARGS__, "Ranks per node: %d", npernode_);
  }

  std::string analysis_str = "";
  if (amr_opts.p2p_enable_matrix_reduce) {
    analysis_str += CollectWithReduce();
    analysis_str += "\n";
  }

  if (amr_opts.p2p_enable_matrix_put) {
    analysis_str += CollectWithPuts();
    analysis_str += "\n";
  }

  return analysis_str;
}

std::string P2PCommCollector::CollectWithReduce() {
  int rv = 0;
  std::string analysis_str = MetricPrintUtils::CommMatrixAnalysisHeader();

  auto analysis = CollectMatrixWithReduce(send_count_, /* is_send= */ true);
  analysis_str +=
      MetricPrintUtils::CountMatrixAnalysisToStr("P2P_Send_Count", analysis);

  analysis = CollectMatrixWithReduce(send_sz_, /* is_send= */ true);
  analysis_str +=
      MetricPrintUtils::SizeMatrixAnalysisToStr("P2P_Send_Size", analysis);

  return analysis_str;
}

std::string P2PCommCollector::CollectWithPuts() {
  int rv = 0;
  std::string analysis_str = MetricPrintUtils::CommMatrixAnalysisHeader();

  auto analysis = CollectMatrixWithPuts(send_count_, /* is_send= */ true);

  analysis_str +=
      MetricPrintUtils::CountMatrixAnalysisToStr("P2P_Send_Count", analysis);

  analysis = CollectMatrixWithPuts(send_sz_, /* is_send= */ true);

  analysis_str +=
      MetricPrintUtils::SizeMatrixAnalysisToStr("P2P_Send_Size", analysis);

  return analysis_str;
}

}  // namespace amr
