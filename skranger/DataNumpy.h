#include "globals.h"
#include "utility.h"
#include "Data.h"

#ifndef DATANUMPY_H_
#define DATANUMPY_H_

namespace ranger {

class DataNumpy: public Data {
public:
  DataNumpy() = default;
  DataNumpy(double* x, double* y, std::vector<std::string> variable_names, size_t num_rows, size_t num_cols, size_t num_cols_y) {
    std::vector<double> xv(x, x + num_cols * num_rows);
    std::vector<double> yv(y, y + num_cols_y * num_rows);
    this->x = xv;
    this->y = yv;
    this->variable_names = variable_names;
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->num_cols_no_snp = num_cols;
  }

  DataNumpy(const DataNumpy&) = delete;
  DataNumpy& operator=(const DataNumpy&) = delete;

  virtual ~DataNumpy() override = default;

  double get_x(size_t row, size_t col) const override {
    size_t col_permuted = col;
    if (col >= num_cols) {
      col = getUnpermutedVarID(col);
      row = getPermutedSampleID(row);
    }

    if (col < num_cols_no_snp) {
      return x[col * num_rows + row];
    } else {
      return getSnp(row, col, col_permuted);
    }
  }

  double get_y(size_t row, size_t col) const override {
    return y[col * num_rows + row];
  }

  void reserveMemory(size_t y_cols) override {
    x.resize(num_cols * num_rows);
    y.resize(y_cols * num_rows);
  }

  void set_x(size_t col, size_t row, double value, bool& error) override {
    x[col * num_rows + row] = value;
  }

  void set_y(size_t col, size_t row, double value, bool& error) override {
    y[col * num_rows + row] = value;
  }

private:
  std::vector<double> x;
  std::vector<double> y;
};

} // namespace ranger

#endif

