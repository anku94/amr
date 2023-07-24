//
// Created by Ankush J on 4/14/23.
//

#include "ilp_callback.h"
#include "lb_policies.h"
#include "policy.h"
#include "tools.h"

#include <gurobi_c++.h>
#include <vector>

namespace {
class ILPSolver {
 public:
  ILPSolver(amr::PolicyOptsILP& opts, const std::vector<double>& cost_list,
            std::vector<int>& rank_list, int nranks)
      : opts_(opts),
        cost_list_(cost_list),
        rank_list_(rank_list),
        nblocks_(cost_list.size()),
        nranks_(nranks),
        assign_vars_(nblocks_, std::vector<GRBVar>(nranks_)) {
    logf(LOG_INFO, "[ILPSolver] Opts: %s", opts_.ToString().c_str());
  }

  int AssignBlocks() {
    try {
      return AssignBlocksInternal();
    } catch (GRBException& e) {
      std::cerr << "Error code = " << e.getErrorCode() << std::endl;
      std::cerr << e.getMessage() << std::endl;
      return e.getErrorCode();
    } catch (...) {
      std::cerr << "Exception during optimization" << std::endl;
      return -1;
    }
  }

 private:
  int AssignBlocksInternal() {
    // Create a Gurobi environment and model
    GRBEnv env = GRBEnv(true);
    InitEnv(env);
    GRBModel model = GRBModel(env);
    InitModel(model);

    std::vector<int> rank_list_ref;
    double max_disorder = InitDecisionVarsWithHeuristic(model, rank_list_ref);

    SolutionCallback cb(assign_vars_);
    model.setCallback(&cb);

    // Create continuous variable load_sum to represent the maximum load on
    // any rank
    GRBVar load_sum = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    GRBLinExpr load_sum_expr(load_sum);

    //      GRBVar load_cnt = model.addVar(0.0, GRB_INFINITY, 0.0,
    //      GRB_CONTINUOUS); GRBLinExpr load_cnt_expr(load_cnt);

    // Set the objective to minimize the maximum load
    //      model.setObjective(load_sum_expr, GRB_MINIMIZE);

    SetUniqueAllocationConstraints(model);

    SetLoadConstraints(model, load_sum);
    //      SetLoadConstraints(model, load_sum, load_cnt);

    //    double disorder_constraint = 0.9 * max_disorder;
    //    logf(LOG_INFO, "Heuristic disorder: %.2lf, Disorder constraint:
    //    %.2lf",
    //         max_disorder, disorder_constraint);

    //      GRBLinExpr loc_score = 0;
    //      SetupLocalityExpr(loc_score);
    //      model.addConstr(loc_score >= disorder_constraint);

    //            GRBQuadExpr loc_score_quad;
    //            SetupQuadraticLocalityExpr(loc_score_quad);
    //            model.addQConstr(loc_score_quad <= 10 * nblocks_);
    GRBLinExpr loc_score;
    SetupGenLocalityExpr3(model, loc_score, rank_list_ref);

    model.setObjectiveN(load_sum_expr, 0, 2, 1, 0, opts_.obj_lb_rel_gap);
    //    model.setObjectiveN(load_sum_expr, 0, 2, 1);
    //      model.setObjectiveN(load_cnt_expr, 1, 1, 1);
    model.setObjectiveN(loc_score, 1, 1, 1);
    //      model.setObjectiveN(load_sum_expr, /* index */ 0, /* priority */
    //      5,
    //                          /* weight */ 1.0, 0.2);
    //      model.setObjectiveN(loc_score_quad, /* index */ 1, /* priority */
    //      4,
    //                          /* weight */ -1.0,
    //                          /* relTol */ 0.00, /* absTol */ 0.0,
    //                          /* objNName */ "loc score");

    // Optimize the model
    model.optimize();
    FillRankList();

    return 0;
  }

  void InitEnv(GRBEnv& env) {
    //    env.set("LogFile", "gurobi.log");
    //      env.set("TimeLimit", "60.0");
    // Illustrative parameter configs below:
    // Illustrative parameter configs below:
    // env.set("Threads", "8");
    // model.getEnv().set(GRB_IntParam_Threads, 8);
    // model.getEnv().set(GRB_DoubleParam_TimeLimit, 45.0);
    env.set(GRB_DoubleParam_MIPGap, opts_.mip_gap);
    env.start();
  }

  void InitModel(GRBModel& model) {
    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    GRBEnv env_lb = model.getMultiobjEnv(0);
    env_lb.set(GRB_DoubleParam_TimeLimit, opts_.obj_lb_time_limit);

    GRBEnv env_loc = model.getMultiobjEnv(1);
    env_loc.set(GRB_DoubleParam_TimeLimit, opts_.obj_loc_time_limit);
  }

  void InitDecisionVars(GRBModel& model) {
    for (int i = 0; i < nblocks_; ++i) {
      for (int j = 0; j < nranks_; ++j) {
        assign_vars_[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      }
    }
  }

  // returns baseline disorder
  double InitDecisionVarsWithHeuristic(GRBModel& model,
                                       std::vector<int>& rank_list_heuristic) {
    amr::LoadBalancePolicies::AssignBlocks(
        amr::LoadBalancePolicy::kPolicyContigImproved, cost_list_,
        rank_list_heuristic, nranks_);

    double rt_avg, rt_max;
    amr::PolicyTools::ComputePolicyCosts(nranks_, cost_list_,
                                         rank_list_heuristic, rt_avg, rt_max);

    double disorder = amr::PolicyTools::GetDisorder(rank_list_heuristic);
    double lin_disorder =
        amr::PolicyTools::GetLinearDisorder(rank_list_heuristic);

    logf(LOG_INFO,
         "Heuristic solution stats.\n"
         "\tdisord:\t%.2lf\n"
         "\tlin_disord:\t%.2lf\n"
         "\trt_avg: \t%.2lf\n"
         "\trt_max:\t%.2lf",
         disorder, lin_disorder, rt_avg, rt_max);

    for (int i = 0; i < nblocks_; ++i) {
      for (int j = 0; j < nranks_; ++j) {
        double flag = (rank_list_heuristic[i] == j ? 1.0 : 0.0);
        assign_vars_[i][j] = model.addVar(0.0, 1.0, flag, GRB_BINARY);
      }
    }

    return lin_disorder;
  }

  void SetUniqueAllocationConstraints(GRBModel& model) {
    for (int i = 0; i < nblocks_; ++i) {
      GRBLinExpr block_sum = 0;
      for (int j = 0; j < nranks_; ++j) {
        block_sum += assign_vars_[i][j];
      }
      model.addConstr(block_sum == 1);
    }
  }

  void SetLoadConstraints(GRBModel& model, GRBVar& sum_limit) {
    for (int j = 0; j < nranks_; ++j) {
      GRBLinExpr load_sum = 0;
      GRBLinExpr load_cnt = 0;
      for (int i = 0; i < nblocks_; ++i) {
        load_sum += cost_list_[i] * assign_vars_[i][j];
        load_cnt += assign_vars_[i][j];
      }
      model.addConstr(load_sum <= sum_limit);
    }
  }

  void SetLoadConstraints(GRBModel& model, GRBVar& sum_limit,
                          GRBVar& count_limit) {
    for (int j = 0; j < nranks_; ++j) {
      GRBLinExpr load_sum = 0;
      GRBLinExpr load_cnt = 0;
      for (int i = 0; i < nblocks_; ++i) {
        load_sum += cost_list_[i] * assign_vars_[i][j];
        load_cnt += assign_vars_[i][j];
      }
      model.addConstr(load_sum <= sum_limit);
      model.addConstr(load_cnt <= count_limit);
    }
  }

  void SetupLocalityExpr(GRBLinExpr& loc_score) {
    for (int i = 0; i < nblocks_; ++i) {
      for (int j = 0; j < nranks_; ++j) {
        loc_score += j * i * assign_vars_[i][j];
      }
    }
  }

  void SetupQuadraticLocalityExpr(GRBQuadExpr& loc_score) {
    GRBLinExpr prev = 0;
    for (int i = 0; i < nblocks_; ++i) {
      GRBLinExpr cur;
      for (int j = 0; j < nranks_; ++j) {
        cur += j * i * assign_vars_[i][j];
      }

      GRBLinExpr diff = cur - prev;
      loc_score += diff * diff;
      prev = cur;
    }
  }

  //
  // Use contig-improved ranks as reference for locality score
  //
  void SetupGenLocalityExpr(GRBModel& model, GRBLinExpr& loc_score,
                            std::vector<int>& rank_list_ref) {
    loc_score = 0;

    for (int bid = 0; bid < nblocks_; bid++) {
      int block_rank_ref = rank_list_ref[bid];
      GRBLinExpr block_rank = 0;
      for (int rank = 0; rank < nranks_; rank++) {
        block_rank += rank * assign_vars_[bid][rank];
      }

      GRBVar block_dist, block_dist_abs;
      model.addConstr(block_dist == block_rank - block_rank_ref);
      model.addGenConstrAbs(block_dist_abs, block_dist);
      loc_score += block_dist_abs;
    }
  }

  //
  // Use array disorder as definition for locality score
  //
  void SetupGenLocalityExpr2(GRBModel& model, GRBLinExpr& loc_score) {
    loc_score = 0;
    GRBLinExpr prev_block_rank = 0;

    for (int bid = 1; bid < nblocks_; bid++) {
      GRBLinExpr cur_block_rank = 0;
      for (int rank = 0; rank < nranks_; rank++) {
        cur_block_rank += rank * assign_vars_[bid][rank];
      }

      GRBVar block_dist =
          model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER);
      GRBVar block_dist_abs = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER);

      model.addConstr(block_dist == cur_block_rank - prev_block_rank);
      model.addGenConstrAbs(block_dist_abs, block_dist);
      loc_score += block_dist_abs;

      prev_block_rank = cur_block_rank;
    }
  }

  //
  // Use contig-improved ranks as reference for locality score
  // Minimize the number of blocks moved, rather than sortedness
  //
  void SetupGenLocalityExpr3(GRBModel& model, GRBLinExpr& loc_score,
                             std::vector<int>& rank_list_ref) {
    loc_score = 0;

    for (int bid = 0; bid < nblocks_; bid++) {
      int block_rank_ref = rank_list_ref[bid];
      GRBLinExpr block_rank = 0;
      for (int rank = 0; rank < nranks_; rank++) {
        block_rank += rank * assign_vars_[bid][rank];
      }

      GRBVar block_dist =
          model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER);
      model.addConstr(block_dist == block_rank - block_rank_ref);
      GRBLinExpr block_dist_expr(block_dist);

      GRBVar block_loc_changed = model.addVar(0.0, 1, 0.0, GRB_BINARY);
      model.addGenConstrIndicator(block_loc_changed, 0, block_dist_expr,
                                  GRB_EQUAL, 0);
      loc_score += block_loc_changed;
    }
  }

  // Fill in the ranklist based on the optimal solution
  void FillRankList() {
    rank_list_.resize(nblocks_, -1);
    for (int i = 0; i < nblocks_; ++i) {
      for (int j = 0; j < nranks_; ++j) {
        if (assign_vars_[i][j].get(GRB_DoubleAttr_X) > 0.5) {
          rank_list_[i] = j;
          break;
        }
      }
    }
  }

  void UseHeuristicSolution(SolutionCallback& cb) {
    std::vector<int> rank_list_heuristic;
    amr::LoadBalancePolicies::AssignBlocks(
        amr::LoadBalancePolicy::kPolicyContigImproved, cost_list_,
        rank_list_heuristic, nranks_);

    cb.SetHeuristicSolution(rank_list_heuristic);
  }

  const amr::PolicyOptsILP opts_;
  const std::vector<double>& cost_list_;
  std::vector<int>& rank_list_;
  int nblocks_;
  int nranks_;

  std::vector<std::vector<GRBVar>> assign_vars_;
};
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksILP(const std::vector<double>& costlist,
                                         std::vector<int>& ranklist, int nranks,
                                         void* opts) {
  PolicyOptsILP opts_obj;
  if (opts) {
    opts_obj = *(PolicyOptsILP*)opts;
  }

  ILPSolver solver(opts_obj, costlist, ranklist, nranks);
  return solver.AssignBlocks();
}
}  // namespace amr
