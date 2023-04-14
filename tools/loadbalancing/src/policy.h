//
// Created by Ankush J on 4/11/23.
//

#pragma once

namespace amr {
enum class Policy {
  kPolicyContiguous,
  kPolicyRoundRobin,
  kPolicySkewed,
  kPolicySPT,
  kPolicyLPT,
  kPolicyILP
};
}