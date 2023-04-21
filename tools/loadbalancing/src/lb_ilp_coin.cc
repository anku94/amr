//
// Created by Ankush J on 4/17/23.
//

// Status: Incomplete. FindCBC was never fully implemented, for complex
// dependency reasons (and Gurobi license was fixed in the meanwhile)

#include "lb_policies.h"

#include <coin/CbcModel.hpp>
// #include <coin/CoinModel.hpp>
#include <coin/OsiClpSolverInterface.hpp>
#include <iostream>
#include <vector>

namespace {
int AssignBlocks(const std::vector<double>& costlist,
                 std::vector<int>& ranklist, int nranks, int maxSeconds = 30) {
  if (costlist.empty() || nranks <= 0) {
    return 1;
  }

  int nblocks = costlist.size();

  // Create a linear programming model
  CoinModel model;

  // Create decision variables: x_{i, j} - 1 if block i is assigned to rank j, 0
  // otherwise
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nranks; ++j) {
      model.setColBounds(i * nranks + j, 0, 1);
      model.setInteger(i * nranks + j);
    }
  }

  // Set constraints - each block must be assigned to exactly one rank
  for (int i = 0; i < nblocks; ++i) {
    CoinBigIndex start = model.getNumElements();
    for (int j = 0; j < nranks; ++j) {
      model.addElement(i, i * nranks + j, 1);
    }
    model.setRowBounds(i, 1, 1);
  }

  // Create artificial variables: one for each rank, representing the total load
  // for that rank
  for (int j = 0; j < nranks; ++j) {
    model.setColBounds(nblocks * nranks + j, 0, COIN_DBL_MAX);
  }

  // Set constraints for artificial variables - each artificial variable should
  // be equal to the total load for that rank
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nranks; ++j) {
      model.addElement(nblocks + j, i * nranks + j, costlist[i]);
      model.addElement(nblocks + j, nblocks * nranks + j, -1);
    }
    model.setRowBounds(nblocks + j, 0, 0);
  }

  // Set the objective function: minimize the maximum load across all ranks
  for (int j = 0; j < nranks; ++j) {
    model.setObjCoeff(nblocks * nranks + j, 1);
  }

  // Create a solver interface
  OsiClpSolverInterface solver;
  solver.loadFromCoinModel(model);

  // Create a mixed-integer programming model
  CbcModel mip(solver);

  // Set the maximum allowed runtime (in seconds)
  mip.setDblParam(CbcModel::CbcMaximumSeconds, maxSeconds);

  // Solve the MIP problem within the given time limit
  mip.branchAndBound();

  // If a solution is found within the given time limit
  if (!mip.isProvenInfeasible()) {
    // Get the best solution found
    const double* solution = mip.getColSolution();

    // Populate ranklist with the best assignment found
    ranklist.resize(nblocks);
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nranks; ++j) {
        if (solution[i * nranks + j] > 0.5) {
          ranklist[i] = j;
        }
      }
    }
    return 0;
  } else {
    // No feasible solution found within the given time limit
    return 1;
  }
}
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksILPCoin(
    const std::vector<double>& costlist, std::vector<int>& ranklist,
    int nranks) {
  return ::AssignBlocks(costlist, ranklist, nranks);
}
}  // namespace amr