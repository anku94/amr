#include "amr/block.h"
#include "comm_mesh.h"
#include "single_ts_trace_reader.h"
#include "tools-common/globals.h"
#include "tools-common/logging.h"
#include "trace_reader.h"

namespace topo {
class MeshGenerator {
 protected:
  MeshGenerator(const DriverOpts &opts) : opts_(opts) {}

  void AddMeshBlock(CommMesh &mesh, std::shared_ptr<MeshBlock> block) {
    mesh.AddBlock(block);
  }

  const DriverOpts &opts_;

 public:
  virtual int GetNumTimesteps() = 0;

  virtual Status GenerateMesh(CommMesh &mesh, int ts) = 0;

  static std::unique_ptr<MeshGenerator> Create(const DriverOpts &opts);

  virtual ~MeshGenerator() = default;
};

class RingMeshGenerator : public MeshGenerator {
 public:
  RingMeshGenerator(const DriverOpts &opts) : MeshGenerator(opts) {}

  int GetNumTimesteps() override { return 1; }

  Status GenerateMesh(CommMesh &mesh, int ts) override;
};

class AllToAllMeshGenerator : public MeshGenerator {
 public:
  AllToAllMeshGenerator(const DriverOpts &opts) : MeshGenerator(opts) {}

  int GetNumTimesteps() override { return 1; }

  Status GenerateMesh(CommMesh &mesh, int ts) override;
};

class SingleTimestepTraceMeshGenerator : public MeshGenerator {
 public:
  SingleTimestepTraceMeshGenerator(const DriverOpts &opts)
      : MeshGenerator(opts), reader_(opts.trace_root) {}

  int GetNumTimesteps() override {
    reader_.Read(Globals::my_rank);
    return 1;
  }

  Status GenerateMesh(CommMesh &mesh, int ts) override;

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

  Status GenerateMesh(CommMesh &mesh, int ts) override;

 private:
  TraceReader reader_;
};
}  // namespace topo