#include <ctime>
#include <string>
#include <unordered_map>

namespace amr {

class MPIAsync {
 public:
  void LogAsyncBegin(const char* key, void* request) {
    ActiveReq req;
    req.key = key;
    req.request = request;
    req.start_time = Now();
    active_reqs_[request] = req;
  }

  int LogMPITestEnd(int flag, void* request, double* elapsed_ms) {
    if (flag == 0) {
      return 0;
    }

    auto it = active_reqs_.find(request);
    if (it == active_reqs_.end()) {
      return 0;
    }

    *elapsed_ms = (Now() - it->second.start_time) / 1e6;
    active_reqs_.erase(it);
    return 1;
  
  }

 private:
  uint64_t Now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
  }

  struct ActiveReq {
    std::string key;
    void* request;
    uint64_t start_time;
  };

  std::unordered_map<void*, ActiveReq> active_reqs_;
};
}  // namespace amr
