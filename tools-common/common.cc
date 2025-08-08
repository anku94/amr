//
// Created by Ankush J on 4/11/22.
//


#include <errno.h>
#include <glog/logging.h>
#include <pdlfs-common/env.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "logging.h"

std::string MeshGenMethodToStr(MeshGenMethod t) {
  switch (t) {
  case MeshGenMethod::Ring:
    return "RING";
  case MeshGenMethod::AllToAll:
    return "ALL_TO_ALL";
  case MeshGenMethod::Dynamic:
    return "DYNAMIC";
  case MeshGenMethod::FromSingleTSTrace:
    return "FROM_SINGLE_TS_TRACE";
  case MeshGenMethod::FromMultiTSTrace:
    return "FROM_MULTI_TS_TRACE";
  }

  return "UNKNOWN";
}