#include "mesh.h"

Mesh1D::Mesh1D(double x_min, double x_max, int n_cells)
    : n_cells_(n_cells), x_min_(x_min), x_max_(x_max),
      edges_(n_cells + 1), centers_(n_cells), widths_(n_cells)
{
    build();
}

void Mesh1D::build()
{
    double dx = (x_max_ - x_min_) / n_cells_;
    for (int i = 0; i <= n_cells_; ++i) {
        edges_[i] = x_min_ + i * dx;
    }
    for (int i = 0; i < n_cells_; ++i) {
        centers_[i] = 0.5 * (edges_[i] + edges_[i + 1]);
        widths_[i] = edges_[i + 1] - edges_[i];
    }
}
