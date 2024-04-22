#include "simple_sim.h"

#include "lb_policies.h"
#include "policy.h"

const char* policy_file = nullptr;

void RunPolicy(const char* policy_name, std::vector<double> const& costlist,
               int nranks) {
  std::vector<int> ranklist;
  std::vector<double> rank_times(nranks, 0);
  double rtavg, rtmax;

  int rv = amr::LoadBalancePolicies::AssignBlocks(policy_name, costlist,
                                                  ranklist, nranks);
  if (rv) {
    ABORT("LB failed");
  }

  amr::PolicyUtils::ComputePolicyCosts(nranks, costlist, ranklist, rank_times,
                                       rtavg, rtmax);

  logv(__LOG_ARGS__, LOG_INFO, "Policy %12s. Avg: %12.0lf, Max: %12.0lf",
       policy_name, rtavg, rtmax);
}

void Run(const char* policy_file) {
  pdlfs::Env* env = pdlfs::Env::Default();
  CSVReader reader(policy_file, env);

  int nlines = 1;
  for (int i = 0; i < nlines; i++) {
    logv(__LOG_ARGS__, LOG_INFO, "-----------\nLine %d", i);

    auto vec = reader.ReadOnce();
    reader.PreviewVector(vec, 10);

    RunPolicy("lpt", vec, 512);
    RunPolicy("hybrid70", vec, 512);
    RunPolicy("hybrid90", vec, 512);
  }
  // auto vec = reader.ReadOnce();
  // reader.PreviewVector(vec, 10);
  //
  // RunPolicy("lpt", vec, 512);
  // RunPolicy("hybrid70", vec, 512);
  // RunPolicy("hybrid90", vec, 512);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <policy_file>\n", argv[0]);
    return 1;
  }

  policy_file = argv[1];
  Run(policy_file);

  return 0;
}
