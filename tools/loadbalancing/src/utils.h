//
// Created by Ankush J on 4/17/23.
//

#pragma once

#include <pdlfs-common/env.h>
#include <regex>
#include <string>

namespace amr {
class Utils {
 public:
  static int EnsureDir(pdlfs::Env* env, const std::string& dir_path) {
    pdlfs::Status s = env->CreateDir(dir_path.c_str());
    if (s.ok()) {
      logf(LOG_INFO, "\t- Created successfully.");
    } else if (s.IsAlreadyExists()) {
      logf(LOG_INFO, "\t- Already exists.");
    } else {
      logf(LOG_ERRO, "Failed to create output directory: %s (Reason: %s)",
           dir_path.c_str(), s.ToString().c_str());
      return -1;
    }

    return 0;
  }

  static std::vector<std::string> FilterByRegex(
      std::vector<std::string>& strings, std::string regex_pattern) {
    std::vector<std::string> matches;
    const std::regex regex_obj(regex_pattern);

    for (auto& s : strings) {
      std::smatch match_obj;
      if (std::regex_match(s, match_obj, regex_obj)) {
        matches.push_back(s);
      }
    }
    return matches;
  }

  static std::vector<std::string> LocateTraceFiles(
      pdlfs::Env* env, const std::string& search_dir) {
    logf(LOG_INFO, "[SimulateTrace] Looking for trace files in: \n\t%s",
         search_dir.c_str());

    std::vector<std::string> all_files;
    env->GetChildren(search_dir.c_str(), &all_files);

    logf(LOG_DBG2, "Enumerating directory: %s", search_dir.c_str());
    for (auto& f : all_files) {
      logf(LOG_DBG2, "- File: %s", f.c_str());
    }

    std::vector<std::string> regex_patterns = {
        R"(prof\.merged\.evt\d+\.csv)",
        R"(prof\.merged\.evt\d+\.mini\.csv)",
        R"(prof\.aggr\.evt\d+\.csv)",
    };

    std::vector<std::string> relevant_files;
    for (auto& pattern : regex_patterns) {
      logf(LOG_DBG2, "Searching by pattern: %s", pattern.c_str());
      relevant_files = FilterByRegex(all_files, pattern);

      for (auto& f : relevant_files) {
        logf(LOG_DBG2, "- Match: %s", f.c_str());
      }

      if (!relevant_files.empty()) break;
    }

    if (relevant_files.empty()) {
      ABORT("no trace files found!");
    }

    std::vector<std::string> all_fpaths;

    for (auto& f : relevant_files) {
      std::string full_path = std::string(search_dir) + "/" + f;
      logf(LOG_INFO, "[ProfSetReader] Adding trace file: %s",
           full_path.c_str());
      all_fpaths.push_back(full_path);
    }

    return all_fpaths;
  }

  static void ExtrapolateCosts(std::vector<double> const& costs_prev,
                               std::vector<int>& refs, std::vector<int>& derefs,
                               std::vector<double>& costs_cur) {
    int nblocks_prev = costs_prev.size();
    int nblocks_cur =
        nblocks_prev + (refs.size() * 7) - (derefs.size() * 7 / 8);

    costs_cur.resize(0);
    std::sort(refs.begin(), refs.end());
    std::sort(derefs.begin(), derefs.end());

    int ref_idx = 0;
    int deref_idx = 0;
    for (int bidx = 0; bidx < nblocks_prev;) {
      if (ref_idx < refs.size() && refs[ref_idx] == bidx) {
        for (int dim = 0; dim < 8; dim++) {
          costs_cur.push_back(costs_prev[bidx]);
        }
        ref_idx++;
        bidx++;
      } else if (deref_idx < derefs.size() && derefs[deref_idx] == bidx) {
        double cost_deref_avg = 0;
        for (int dim = 0; dim < 8; dim++) {
          cost_deref_avg += costs_prev[bidx + dim];
        }
        cost_deref_avg /= 8;
        costs_cur.push_back(cost_deref_avg);
        deref_idx += 8;
        bidx += 8;
      } else {
        costs_cur.push_back(costs_prev[bidx]);
        bidx++;
      }
    }

    assert(costs_cur.size() == nblocks_cur);
  }
};
}  // namespace amr