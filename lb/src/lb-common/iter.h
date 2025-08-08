#include <map>

namespace amr {
class IterationTracker {
 public:
   IterationTracker() : last_cost_(-1) {}

   void Clear() { costs_seen_.clear(); last_cost_ = -1; }

   void LogCost(double cost) {
     costs_seen_[cost] = true;
     last_cost_ = cost;
   }

   bool ShouldStop(double cost) {
     if (costs_seen_.find(cost) != costs_seen_.end()) {
       if (cost < last_cost_) {
         return true;
       }
     }

     LogCost(cost);
     return false;
   }
 private:
   std::map<int, bool> costs_seen_;
   int last_cost_;
};
}  // namespace amr
