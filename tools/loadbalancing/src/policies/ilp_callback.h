//
// Created by Ankush J on 7/21/23.
//

#pragma once

#include "common.h"
#include "gurobi_c++.h"
#include "tools.h"

#include <vector>

namespace {
class SolutionCallback : public GRBCallback {
 public:
  SolutionCallback(std::vector<std::vector<GRBVar>>& vars) : vars_(vars) {}

  void SetHeuristicSolution(std::vector<int> const& rank_list) {
    rank_list_heuristic_ = rank_list;
  }

 protected:
  void callback() override {
    try {
      if (where == GRB_CB_MIPSOL || where == GRB_CB_MULTIOBJ) {
        LogSolution();
      } else {
        //        logf(LOG_INFO, "Callback type: other. Where: %d", where);
      }
    } catch (GRBException& e) {
      std::cerr << "Error number: " << e.getErrorCode() << std::endl;
      std::cerr << e.getMessage() << std::endl;
    } catch (...) {
      std::cerr << "Error during callback" << std::endl;
    }
  }

 private:
  void LogSolution() {
    if (vars_.empty()) return;

    int nblocks = vars_.size();
    int nranks = vars_[0].size();

    std::vector<int> rank_list(nblocks, -1);

    for (int i = 0; i < nblocks; i++) {
      for (int j = 0; j < nranks; j++) {
        if (getSolution(vars_[i][j]) > 0.5) {
          rank_list[i] = j;
        }
      }
    }

    double disorder = amr::PolicyTools::GetDisorder(rank_list);
    double lin_disorder = amr::PolicyTools::GetLinearDisorder(rank_list);
    logf(LOG_INFO, "Solution callback. Disorder: %.2lf, Lin_Disorder: %.2lf",
         disorder, lin_disorder);
  }

  void LoadHeuristicSolution() {
    if (rank_list_heuristic_.empty()) return;

    logf(LOG_INFO, "[ALERT] ILP CALLBACK: Loading heuristic solution.");

    int nblocks = vars_.size();
    int nranks = vars_[0].size();

    for (int i = 0; i < nblocks; i++) {
      for (int j = 0; j < nranks; j++) {
        setSolution(vars_[i][j], (double)(rank_list_heuristic_[i] == j));
      }
    }

    rank_list_heuristic_.clear();
  }

  std::vector<std::vector<GRBVar>>& vars_;
  std::vector<int> rank_list_heuristic_;
};
}  // namespace
