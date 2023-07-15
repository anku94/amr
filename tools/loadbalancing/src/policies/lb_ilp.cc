//
// Created by Ankush J on 4/14/23.
//

#include "lb_policies.h"

#include <gurobi_c++.h>
#include <vector>

namespace {
int AssignBlocks(const std::vector<double>& costlist,
                 std::vector<int>& ranklist, int nranks) {
  try {
    // Create a Gurobi environment and model
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "gurobi.log");
    env.set("TimeLimit", "45.0");
    // Illustrative parameter configs below:
    // env.set("Threads", "8");
    // model.getEnv().set(GRB_IntParam_Threads, 8);
    // model.getEnv().set(GRB_DoubleParam_TimeLimit, 45.0);
    env.start();

    GRBModel model = GRBModel(env);

    int nblocks = costlist.size();

    // Create binary decision variables x[i][j]
    std::vector<std::vector<GRBVar>> x(nblocks, std::vector<GRBVar>(nranks));
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nranks; ++j) {
        x[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      }
    }

    // Create continuous variable y to represent the maximum load on any rank
    GRBVar y = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    GRBLinExpr ylexpr(y);

    // Set the objective to minimize the maximum load
    model.setObjective(ylexpr, GRB_MINIMIZE);

    // Add constraints
    for (int i = 0; i < nblocks; ++i) {
      GRBLinExpr block_sum = 0;
      for (int j = 0; j < nranks; ++j) {
        block_sum += x[i][j];
      }
      model.addConstr(block_sum ==
                      1);  // Each block is assigned to exactly one rank
    }

    for (int j = 0; j < nranks; ++j) {
      GRBLinExpr load_sum = 0;
      for (int i = 0; i < nblocks; ++i) {
        load_sum += costlist[i] * x[i][j];
      }
      model.addConstr(load_sum <=
                      y);  // Load of each rank is less than or equal to y
    }

    // Optimize the model
    model.optimize();

    // Fill in the ranklist based on the optimal solution
    ranklist.resize(nblocks, -1);
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nranks; ++j) {
        if (x[i][j].get(GRB_DoubleAttr_X) > 0.5) {
          ranklist[i] = j;
          break;
        }
      }
    }

    return 0;
  } catch (GRBException& e) {
    std::cerr << "Error code = " << e.getErrorCode() << std::endl;
    std::cerr << e.getMessage() << std::endl;
    return e.getErrorCode();
  } catch (...) {
    std::cerr << "Exception during optimization" << std::endl;
    return -1;
  }
}
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksILP(const std::vector<double>& costlist,
                                         std::vector<int>& ranklist,
                                         int nranks) {
  return ::AssignBlocks(costlist, ranklist, nranks);
}
}  // namespace amr
