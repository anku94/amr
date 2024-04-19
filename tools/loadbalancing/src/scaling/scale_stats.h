#include "common.h"
#include "policy.h"
#include "tabular_data.h"

namespace amr {
class ScaleSimRow : public TableRow {
 private:
  const std::string policy_;
  const int nblocks_;
  const int nranks_;
  const double iter_time_;
  const double rt_avg_;
  const double rt_max_;
  const double loc_cost_;

  std::vector<std::string> header = {"policy",      "nblocks", "nranks",
                                     "iter_us",     "rt_avg",  "rt_max",
                                     "loc_cost_pct"};

 public:
  ScaleSimRow(const std::string policy_name, int nblocks, int nranks,
              double iter_time, double rt_avg, double rt_max, double loc_cost)
      : policy_(policy_name),
        nblocks_(nblocks),
        nranks_(nranks),
        iter_time_(iter_time),
        rt_avg_(rt_avg),
        rt_max_(rt_max),
        loc_cost_(loc_cost) {}

  std::vector<std::string> GetHeader() const override { return header; }

  std::vector<std::string> GetData() const override {
    return {policy_,
            std::to_string(nblocks_),
            std::to_string(nranks_),
            FixedWidthString(iter_time_, 1),
            FixedWidthString(rt_avg_, 0),
            FixedWidthString(rt_max_, 0),
            FixedWidthString(loc_cost_, 0)};
  }

  static std::string FixedWidthString(double d, int n) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(n) << d;
    return ss.str();
  }
};
};  // namespace amr
