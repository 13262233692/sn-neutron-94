#pragma once

#include <vector>

class Mesh1D {
public:
    Mesh1D(double x_min, double x_max, int n_cells);

    int n_cells() const { return n_cells_; }
    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }

    const std::vector<double>& edges() const { return edges_; }
    const std::vector<double>& centers() const { return centers_; }
    const std::vector<double>& widths() const { return widths_; }

    double edge(int i) const { return edges_[i]; }
    double center(int i) const { return centers_[i]; }
    double width(int i) const { return widths_[i]; }

private:
    int n_cells_;
    double x_min_;
    double x_max_;
    std::vector<double> edges_;
    std::vector<double> centers_;
    std::vector<double> widths_;

    void build();
};
