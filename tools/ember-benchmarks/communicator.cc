#include "communicator.h"

void Communicator::DoIteration() {
  int req_count = 0;
  DoFace(req_count);
  DoEdge(req_count);

  DLOG_IF(INFO, rank_ == 0)
      << "Rank " << rank_ << ": " << req_count << " requests";

  MPI_Waitall(req_count, request_, status_);
  // DoPoll(req_count);
  AssertAllSuccess(status_, req_count);
}
void Communicator::DoPoll(int num_requests) {
  int completed[num_requests];
  for (int i = 0; i < num_requests; i++) {
    completed[i] = 0;
  }

  int completed_count = 0;

  while (1) {
    int flag;
    MPI_Status status;
    for (int i = 0; i < num_requests; i++) {
      if (!completed[i]) {
        MPI_Test(&request_[i], &flag, &status);
        if (flag) {
          completed[i] = 1;
          completed_count++;
        }
      }
    }

    if (completed_count == num_requests) {
      DLOG(INFO) << "Rank " << rank_ << ": All requests completed";
      return; // All requests are completed, exit the loop
    } else {
      std::string pending_req_str = "";
      for (int i = 0; i < num_requests; i++) {
        if (!completed[i]) {
          pending_req_str += std::to_string(i) + ", ";
        }
      }

      LOG(INFO) << "Rank " << rank_ << ": " << completed_count << "/"
                << num_requests
                << " requests completed. Pending: " + pending_req_str;
    }
    sleep(2); // Wait for 1 second before polling again
  }
}

void Communicator::DoFace(int &req_count) {
  auto face_neighbors = nrg_.GetFaceNeighbors();
  int face_tags[] = {1000, 1000, 2000, 2000, 4000, 4000};
  for (size_t dest_idx = 0; dest_idx < face_neighbors.size(); dest_idx++) {
    int dest = face_neighbors[dest_idx];

    DLOG(INFO) << "[F] Rank " << rank_ << ": [" << dest_idx << "] dest "
               << dest;

    if (dest == -1) {
      continue;
    }

    auto &buf = buffer_suite_.face_buffers_[dest_idx];
    auto msg_tag = face_tags[dest_idx];
    DoCommSingle(buf, dest, msg_tag, req_count);
  }
}

void Communicator::DoEdge(int &req_count) {
  auto edge_neighbors = nrg_.GetEdgeNeighbors();
  int edge_tag = 8000;

  for (size_t dest_idx = 0; dest_idx < edge_neighbors.size(); dest_idx++) {
    int dest = edge_neighbors[dest_idx];

    DLOG(INFO) << "[E] Rank " << rank_ << ": [" << dest_idx << "] dest "
               << dest;

    if (dest == -1) {
      continue;
    }

    auto &buf = buffer_suite_.edge_buffers_[dest_idx];
    auto msg_size = edge_tag;
    DoCommSingle(buf, dest, msg_size, req_count);
  }
}

void Communicator::DoCommSingle(BufferPair &buffer_pair, int dest, int msg_tag,
                                int &req_count) {
  if (rank_ == 0) {
    DLOG(INFO) << "Rank " << rank_ << ": sending " << msg_tag << " to " << dest;
  }

  MPI_Isend(buffer_pair.send_buffer_.data(), buffer_pair.send_buffer_.size(),
            MPI_DOUBLE, dest, msg_tag, MPI_COMM_WORLD, &request_[req_count++]);
  MPI_Irecv(buffer_pair.recv_buffer_.data(), buffer_pair.recv_buffer_.size(),
            MPI_DOUBLE, dest, msg_tag, MPI_COMM_WORLD, &request_[req_count++]);

  int snd_sz = buffer_pair.send_buffer_.size() * sizeof(double);
  int recv_sz = buffer_pair.recv_buffer_.size() * sizeof(double);
  bytes_total_ += snd_sz + recv_sz;
}

void Communicator::LogMPIStatus(MPI_Status *all_status, int n) {
  if (rank_ != 0)
    return;

  int nsuccess = 0;
  for (int i = 0; i < n; i++) {
    if (all_status[i].MPI_ERROR == MPI_SUCCESS) {
      nsuccess++;
    }
  }

  printf("MPI_Status: %d/%d success\n", nsuccess, n);
}

void Communicator::AssertAllSuccess(MPI_Status *all_status, int n) {
  for (int i = 0; i < n; i++) {
    if (all_status[i].MPI_ERROR != MPI_SUCCESS) {
      printf("MPI_Status: %d/%d success\n", i, n);
      exit(1);
    }
  }
}
