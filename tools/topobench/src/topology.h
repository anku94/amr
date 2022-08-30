//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "block.h"
#include "common.h"
#include "graph.h"
#include "trace_reader.h"

class Topology {
 public:
  Topology(const DriverOpts& opts) : opts_(opts), reader_(opts.trace_root) {}

  Status GenerateMesh(const DriverOpts& opts, Mesh& mesh, int ts);

  int GetNumTimesteps() {
    reader_.Read();
    return std::max(reader_.GetNumTimesteps(), opts_.comm_rounds);
  }

 private:
  Status GenerateMeshRing(Mesh& mesh, int ts) const;

  Status GenerateMeshAllToAll(Mesh& mesh, int ts) const;

  Status GenerateMeshDynamic(Mesh& mesh, int ts) {
    Status s = Status::OK;

    logf(LOG_INFO, "Generating Mesh: Dynamic. NOT IMPLEMENTED YET.");

    return s;
  }

  Status GenerateMeshFromTrace(Mesh& mesh, int ts);

  const DriverOpts& opts_;
  TraceReader reader_;
};