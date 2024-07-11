#pragma once

#include <sys/stat.h>

#include <string>
#include <unordered_map>

#include "globals.h"

namespace amr {
// fwd decl
class ConfigUtils;

class ConfigParser {
 private:
  explicit ConfigParser(const char* file_path) { Initialize(file_path); }

  std::unordered_map<std::string, std::string> params_;
  friend class ConfigUtils;

  template <typename T>
  T GetParamOrDefault(const std::string& key, const T& default_val) {
    auto it = params_.find(key);
    if (it == params_.end()) {
      logv(__LOG_ARGS__, LOG_DBG2,
           "Key %s not found in config file, using default value\n",
           key.c_str());
      return default_val;
    }

    return Convert<T>(it->second);
  }

  static std::string Strip(const std::string& str);

  void Initialize(const char* file_path);

  template <typename T>
  T Convert(const std::string& s) {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
};

class ConfigUtils {
 public:
  template <typename T>
  static T GetParamOrDefault(const std::string& key, const T& default_val) {
    Initialize();

    if (!Globals::config) {
      logv(__LOG_ARGS__, LOG_DBUG, "Globals.config is not set\n");
      return default_val;
    }

    return Globals::config->GetParamOrDefault<T>(key, default_val);
  }

 private:
  static void Initialize() {
    if (Globals::config != nullptr) {
      logv(__LOG_ARGS__, LOG_DBUG, "Globals.config already set\n");
      return;
    }

    // Check if env var LB_CONFIG_PATH is set
    const char* env_var = std::getenv("DISTRIB_CONFIG_FPATH");
    if (env_var == nullptr) {
      logv(__LOG_ARGS__, LOG_DBUG, "DISTRIB_CONFIG_FPATH not set\n");
      return;
    }

    // Check if a file exists at the path
    struct stat statbuf;
    if (stat(env_var, &statbuf) != 0) {
      logv(__LOG_ARGS__, LOG_DBUG, "Config file %s does not exist\n", env_var);
      return;
    }

    if (S_ISREG(statbuf.st_mode)) {
      // make_unique does not work with private constructors
      Globals::config =
          std::unique_ptr<ConfigParser>(new ConfigParser(env_var));
    } else {
      logv(__LOG_ARGS__, LOG_DBUG, "Config file %s is not a regular file\n",
           env_var);
    }
  }
};

// specialization of templated functions
template <>
int ConfigParser::Convert(const std::string& s);

template <>
double ConfigParser::Convert(const std::string& s);

template <>
std::string ConfigParser::Convert(const std::string& s);
}  // namespace amr
