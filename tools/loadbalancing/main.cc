#include "policy_sim.h"

#include <cstdio>

void run() {
  amr::PolicySimOptions options;
  options.env = pdlfs::Env::Default();
  options.prof_dir = "/Users/schwifty/Repos/amr-data/20230410-lbdev";
  options.policy = amr::Policy::kPolicyContiguous;

  amr::PolicySim sim(options);
  sim.Run();
}

int main() {
  run();
  return 0;
}
