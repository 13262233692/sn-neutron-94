#pragma once

#include "mesh.h"
#include "xsdata.h"
#include <vector>

class DSA {
public:
    DSA(const Mesh1D& mesh, const XSData& xs, double omega = 1.0);

    std::vector<double> solve(const std::vector<double>& residual) const;

    void set_omega(double omega) { omega_ = omega; }
    double omega() const { return omega_; }

private:
    int n_cells_;
    double omega_;
    std::vector<double> D_;
    std::vector<double> sigma_a_;
    std::vector<double> sigma_s_;
    std::vector<double> dx_;

    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<double> c_;

    void build_matrix();
    std::vector<double> thomas_solve(const std::vector<double>& rhs) const;
};
