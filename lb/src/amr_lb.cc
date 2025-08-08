#include "amr_lb.h"

#include "lb-common/lb_policies.h"
#include "lb-common/policy_utils.h"

namespace amr {
namespace lb {
int LoadBalance::AssignBlocks(PlacementArgs args) {
  auto& policy = PolicyUtils::GetPolicy(args.policy_name.c_str());
  return LoadBalancePolicies::AssignBlocks(policy, args.costlist, args.ranklist,
                                           args.nranks);
}

int LoadBalance::AssignBlocksMpi(PlacementArgsMpi args) {
  auto& pin = args.stdargs;
  // This interface includes caching, and is not satistfactory, but
  // retaining for compatibility for now
  return LoadBalancePolicies::AssignBlocksCached(
      pin.policy_name.c_str(), pin.costlist, pin.ranklist, args.nranks,
      args.my_rank, args.comm);
}
}  // namespace lb
}  // namespace amr