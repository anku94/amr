//
// Created by Ankush J on 11/30/23.
//

#pragma once

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>

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

class EmberUtils {
 public:
  static inline int StrToInt(const char* str) {
    return static_cast<int>(strtoimax(str, nullptr, 10));
  }

  static void PrintHelp() {
    const char* help_str =
        "Usage: ./ember-benchmarks [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -a, --nx=Nx            Number of cells in x-direction\n"
        "  -b, --ny=Ny            Number of cells in y-direction\n"
        "  -c, --nz=Nz            Number of cells in z-direction\n"
        "  -p, --pex=PEX          Number of ranks in x-direction\n"
        "  -q, --pey=PEY          Number of ranks in y-direction\n"
        "  -r, --pez=PEZ          Number of ranks in z-direction\n"
        "  -i, --iterations=ITER  Number of iterations\n"
        "  -v, --vars=VARS        Number of variables\n"
        "  -s, --sleep=SLEEP      Sleep time in nanoseconds\n"
        "  -m, --map=MAP          Mapping file\n"
        "\n"
        "Example:\n"
        "  ./ember-benchmarks -a 10 -b 10 -c 10 -p 2 -q 2 -r 2 -i 100"
        " -v 1 -i 1000 -m map.txt";
    printf("%s\n", help_str);
  }

  static EmberOpts ParseOptions(int argc, char* argv[]) {
    EmberOpts opts;

    int c;
    int option_index = 0;

    static struct option long_options[] = {
        {"nx", required_argument, nullptr, 'a'},
        {"ny", required_argument, nullptr, 'b'},
        {"nz", required_argument, nullptr, 'c'},
        {"pex", required_argument, nullptr, 'p'},
        {"pey", required_argument, nullptr, 'q'},
        {"pez", required_argument, nullptr, 'r'},
        {"iterations", required_argument, nullptr, 'i'},
        {"vars", required_argument, nullptr, 'v'},
        {"sleep", required_argument, nullptr, 's'},
        {"map", required_argument, nullptr, 'm'},
        {nullptr, 0, nullptr, 0}};

    while ((c = getopt_long(argc, argv, "a:b:c:p:q:r:i:v:s:m:", long_options,
                            &option_index)) != -1) {
      switch (c) {
        case 'a':
          opts.nx = StrToInt(optarg);
          break;
        case 'b':
          opts.ny = StrToInt(optarg);
          break;
        case 'c':
          opts.nz = StrToInt(optarg);
          break;
        case 'p':
          opts.pex = StrToInt(optarg);
          break;
        case 'q':
          opts.pey = StrToInt(optarg);
          break;
        case 'r':
          opts.pez = StrToInt(optarg);
          break;
        case 'i':
          opts.iterations = StrToInt(optarg);
          break;
        case 'v':
          opts.vars = StrToInt(optarg);
          break;
        case 's':
          opts.sleep = StrToInt(optarg);
          break;
        case 'm':
          opts.map = optarg;
          break;
        default:
          PrintHelp();
          abort();
      }
    }

    return opts;
  }
};
