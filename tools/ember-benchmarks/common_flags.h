#pragma once

#include <gflags/gflags.h>

struct EmberOpts {
  int nx, ny, nz;
  int pex, pey, pez;
  int iterations;
  int vars;
  int sleep;
  const char* map;

  EmberOpts()
      : nx(10),
        ny(10),
        nz(10),
        pex(4),
        pey(4),
        pez(4),
        iterations(10),
        vars(1),
        sleep(1000),
        map("<undefined>") {}

  std::string ToString() const {
    char str_buf[1024];
    snprintf(str_buf, 1024,
             "[EmberOpts]\n\tnx=%d, ny=%d, nz=%d\n"
             "\tpex=%d, pey=%d, pez=%d\n"
             "\titerations=%d, vars=%d, sleep=%d\n"
             "\tmap=%s\n",
             nx, ny, nz, pex, pey, pez, iterations, vars, sleep,
             map == nullptr ? "<nullptr>" : map);

    return {str_buf};
  }
};

// Define your flags
DEFINE_int32(nx, 0, "Number of cells in x-direction");
DEFINE_int32(ny, 0, "Number of cells in y-direction");
DEFINE_int32(nz, 0, "Number of cells in z-direction");
DEFINE_int32(pex, 0, "Number of ranks in x-direction");
DEFINE_int32(pey, 0, "Number of ranks in y-direction");
DEFINE_int32(pez, 0, "Number of ranks in z-direction");
DEFINE_int32(iterations, 0, "Number of iterations");
DEFINE_int32(vars, 0, "Number of variables");
DEFINE_int32(sleep, 0, "Sleep time in nanoseconds");
DEFINE_string(map, "", "Mapping file");

static bool ValidateMapFile(const char* flagname, const std::string& value) {
  if (value.empty()) {
    printf("Map file must be specified.\n");
    return false;
  }

  // Check if file exists, and is a valid file
  FILE* f = fopen(value.c_str(), "r");
  if (f == nullptr) {
    printf("Map file %s does not exist.\n", value.c_str());
    return false;
  }
  fclose(f);
  return true;
}

// DEFINE_validator(map, &ValidateMapFile);

class EmberUtils {
 public:
  static EmberOpts ParseOptions(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    EmberOpts opts;
    opts.nx = FLAGS_nx;
    opts.ny = FLAGS_nx;
    opts.nz = FLAGS_nx;
    opts.pex = FLAGS_pex;
    opts.pey = FLAGS_pey;
    opts.pez = FLAGS_pez;
    opts.iterations = FLAGS_iterations;
    opts.vars = FLAGS_vars;
    opts.sleep = FLAGS_sleep;
    opts.map = FLAGS_map.c_str();
    // opts.map = nullptr;

    return opts;
  }
};
