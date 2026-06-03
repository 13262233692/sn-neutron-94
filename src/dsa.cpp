#include "dsa.h"
#include <cmath>
#include <algorithm>

DSA::DSA(const Mesh1D& mesh, const XSData& xs, double omega)
    : n_cells_(mesh.n_cells()), omega_(omega),
      D_(n_cells_), sigma_a_(n_cells_), sigma_s_(n_cells_), dx_(n_cells_),
      a_(n_cells_), b_(n_cells_), c_(n_cells_)
{
    for (int i = 0; i < n_cells_; ++i) {
        dx_[i] = mesh.width(i);
        double sigma_t = xs.sigma_t(i);
        double sigma_s = xs.sigma_s(i);
        sigma_s_[i] = sigma_s;
        sigma_a_[i] = sigma_t - sigma_s;
        D_[i] = 1.0 / (3.0 * sigma_t);
    }

    build_matrix();
}

void DSA::build_matrix()
{
    std::fill(a_.begin(), a_.end(), 0.0);
    std::fill(b_.begin(), b_.end(), 0.0);
    std::fill(c_.begin(), c_.end(), 0.0);

    for (int i = 0; i < n_cells_; ++i) {
        if (i > 0) {
            double D_face = 2.0 * D_[i - 1] * D_[i] / (D_[i - 1] * dx_[i] + D_[i] * dx_[i - 1] + 1e-30);
            double coeff = D_face / (0.5 * (dx_[i - 1] + dx_[i]));
            a_[i] = -coeff;
            b_[i] += coeff;
        } else {
            double D_face = D_[0] / (0.5 * dx_[0] + 1e-30);
            b_[0] += D_face / (0.5 * dx_[0]);
        }

        if (i < n_cells_ - 1) {
            double D_face = 2.0 * D_[i] * D_[i + 1] / (D_[i] * dx_[i + 1] + D_[i + 1] * dx_[i] + 1e-30);
            double coeff = D_face / (0.5 * (dx_[i] + dx_[i + 1]));
            c_[i] = -coeff;
            b_[i] += coeff;
        } else {
            double D_face = D_[n_cells_ - 1] / (0.5 * dx_[n_cells_ - 1] + 1e-30);
            b_[n_cells_ - 1] += D_face / (0.5 * dx_[n_cells_ - 1]);
        }

        b_[i] += sigma_a_[i];
    }
}

std::vector<double> DSA::solve(const std::vector<double>& residual) const
{
    std::vector<double> rhs(n_cells_, 0.0);
    for (int i = 0; i < n_cells_; ++i) {
        rhs[i] = sigma_s_[i] * residual[i];
    }

    std::vector<double> correction = thomas_solve(rhs);

    for (int i = 0; i < n_cells_; ++i) {
        correction[i] *= omega_;
    }

    return correction;
}

std::vector<double> DSA::thomas_solve(const std::vector<double>& rhs) const
{
    int n = n_cells_;
    std::vector<double> cp(n, 0.0);
    std::vector<double> dp(n, 0.0);
    std::vector<double> x(n, 0.0);

    cp[0] = c_[0] / b_[0];
    dp[0] = rhs[0] / b_[0];

    for (int i = 1; i < n; ++i) {
        double denom = b_[i] - a_[i] * cp[i - 1];
        if (std::abs(denom) < 1e-30) denom = 1e-30;
        cp[i] = c_[i] / denom;
        dp[i] = (rhs[i] - a_[i] * dp[i - 1]) / denom;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }

    return x;
}
