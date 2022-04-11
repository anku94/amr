#include <iostream>

#include "driver.h"

int main(int argc, char* argv[]) {
  DriverOpts opts;
  Driver driver(opts);
  driver.Run(argc, argv);
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
