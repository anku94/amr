#pragma once

#include "common.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace amr {
class TableRow {
 public:
  virtual ~TableRow() = default;
  virtual std::vector<std::string> GetHeader() const = 0;
  virtual std::vector<std::string> GetData() const = 0;
};

class TabularData {
 private:
  std::vector<std::shared_ptr<TableRow>> rows;
  const std::string delimiter_ = ", ";
  const int min_col_w_ = 10;
  std::vector<int> col_w_;

 public:
  TabularData() = default;

  void addRow(const std::shared_ptr<TableRow>& row) {
    auto data = row->GetData();
    col_w_.resize(data.size(), min_col_w_);
    for (size_t i = 0; i < data.size(); ++i) {
      auto& cell = data[i];
      if (cell.size() > col_w_[i]) {
        col_w_[i] = cell.size() + 2;
      }
    }
    rows.push_back(row);
  }

  std::string toCSV() const {
    std::stringstream ss;
    if (!rows.empty()) {
      auto header = rows.front()->GetHeader();
      for (size_t i = 0; i < header.size(); ++i) {
        ss << header[i] << (i < header.size() - 1 ? "," : "\n");
      }
    }
    for (const auto& row : rows) {
      auto data = row->GetData();
      for (size_t i = 0; i < data.size(); ++i) {
        ss << data[i] << (i < data.size() - 1 ? "," : "\n");
      }
    }
    return ss.str();
  }

  void emitTable(std::ostream& out) const {
    size_t col_w_sum = std::accumulate(col_w_.begin(), col_w_.end(), 0);
    if (!rows.empty()) {
      out << std::left;
      auto header = rows.front()->GetHeader();
      for (size_t i = 0; i < header.size(); ++i) {
        out << std::setw(col_w_[i]) << header[i];
      }
      out << std::endl;
      out << std::string(col_w_sum, '-') << std::endl;
    }
    for (const auto& row : rows) {
      auto data = row->GetData();
      for (size_t i = 0; i < data.size(); ++i) {
        out << std::setw(col_w_[i]) << data[i];
      }
      out << std::endl;
    }
  }
};
}  // namespace amr