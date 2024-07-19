#pragma once

#include <unistd.h>

#include <cassert>
#include <cstring>
#include <string>

//
// PerfManager: class to toggle perf data collection via its control hooks
// Borrowed from: https://pramodkumbhar.com
// Usage:
// pmon.pause() to send disable
// pmon.resume() to send enable
//

class PerfManager {
  // control and ack fifo from perf
  int ctl_fd_ = -1;
  int ack_fd_ = -1;

  // if perf is enabled
  bool enable_ = false;

  // commands and acks to/from perf
  static constexpr const char *enable_cmd = "enable";
  static constexpr const char *disable_cmd = "disable";
  static constexpr const char *ack_cmd = "ack\n";

  // send command to perf via fifo and confirm ack
  void send_command(const char *command) {
    if (enable_) {
      write(ctl_fd_, command, strlen(command));
      char ack[5];
      read(ack_fd_, ack, 5);
      assert(strcmp(ack, ack_cmd) == 0);
    }
  }

 public:
  PerfManager() {
    // setup fifo file descriptors
    char *ctl_fd_env = getenv("PERF_CTL_FD");
    char *ack_fd_env = getenv("PERF_ACK_FD");
    if (ctl_fd_env && ack_fd_env) {
      enable_ = true;
      ctl_fd_ = std::stoi(ctl_fd_env);
      ack_fd_ = std::stoi(ack_fd_env);
    }
  }

  void Pause() { send_command(disable_cmd); }

  void Resume() { send_command(enable_cmd); }
};
