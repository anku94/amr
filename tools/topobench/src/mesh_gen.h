#include "common.h"
#include "globals.h"
#include "mesh.h"
#include "single_ts_trace_reader.h"
#include "trace_reader.h"

class MeshGenerator {
protected:
  MeshGenerator(const DriverOpts &opts) : opts_(opts) {}

  void AddMeshBlock(Mesh &mesh, std::shared_ptr<MeshBlock> block) {
    mesh.AddBlock(block);
  }

  const DriverOpts &opts_;

public:
  virtual int GetNumTimesteps() = 0;

  virtual Status GenerateMesh(Mesh &mesh, int ts) = 0;

  static std::unique_ptr<MeshGenerator> Create(const DriverOpts &opts);
};

class RingMeshGenerator : public MeshGenerator {
public:
  RingMeshGenerator(const DriverOpts &opts) : MeshGenerator(opts) {}

  int GetNumTimesteps() override { return 1; }

  Status GenerateMesh(Mesh &mesh, int ts) override;
};

class AllToAllMeshGenerator : public MeshGenerator {
public:
  AllToAllMeshGenerator(const DriverOpts &opts) : MeshGenerator(opts) {}

  int GetNumTimesteps() override { return 1; }

  Status GenerateMesh(Mesh &mesh, int ts) override;
};

class SingleTimestepTraceMeshGenerator : public MeshGenerator {
public:
  SingleTimestepTraceMeshGenerator(const DriverOpts &opts)
      : MeshGenerator(opts), reader_(opts.trace_root) {}

  int GetNumTimesteps() override {
    reader_.Read(Globals::my_rank);
    return 1;
  }

  Status GenerateMesh(Mesh &mesh, int ts) override;

private:
  SingleTimestepTraceReader reader_;
};

class MultiTimestepTraceMeshGenerator : public MeshGenerator {
public:
  MultiTimestepTraceMeshGenerator(const DriverOpts &opts)
      : MeshGenerator(opts), reader_(opts.trace_root) {}

  int GetNumTimesteps() override {
    reader_.Read(Globals::my_rank);
    return reader_.GetNumTimesteps();
  }

  Status GenerateMesh(Mesh &mesh, int ts) override;

private:
  TraceReader reader_;
};
