//
// Created by Ankush J on 4/8/22.
//

#pragma once

class BvarComm {
  void SendBoundaryBuffers() {
    // get block ptr
    // for each neighbour block (logical) as nb,
    // load buf pointed by nb.bufid to bd_var_.send (BufArray1D type)
    // fence()
    // if local() CopyVariableBufferSameProcess()
    // else MPI_Start()
  }

  void ReceiveBoundaryBuffers() {
    // get block ptr
    // for each neighbor as nb
    // if flag == arrived conitnue;
    // if neighbor on same rank, skip
    // else

    //    PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
    //                                   MPI_STATUS_IGNORE));
    //    PARTHENON_MPI_CHECK(
    //        MPI_Test(&(bd_var_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE));
    //    if (!static_cast<bool>(test)) {
    //      bflag = false;
    //      continue;
    //    }
    //    bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;
  }
};