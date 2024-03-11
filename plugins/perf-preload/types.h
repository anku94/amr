#include <unordered_map>
#include <stack>

namespace amr {
class Metric;
typedef std::unordered_map<std::string, Metric> MetricMap;
typedef std::pair<std::string, uint64_t> StackOpenPair;
typedef std::stack<StackOpenPair> ProfStack;
typedef std::unordered_map<std::string, ProfStack> StackMap;
}
