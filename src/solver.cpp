#include "solver.h"
#include <cmath>
#include <algorithm>
#include <numeric>

SNSolver::SNSolver(const GaussLegendreQuadrature& quad,
                   const Mesh1D& mesh,
                   const XSData& xs,
                   double tol,
                   int max_iter)
    : quad_(quad), mesh_(mesh), xs_(xs),
      tol_(tol), max_iter_(max_iter),
      left_bc_(0.0), right_bc_(0.0),
      keff_(1.0), dsa_enabled_(true),
      dsa_(std::make_unique<DSA>(mesh, xs, 1.0))
{
    int nc = mesh_.n_cells();
    int nd = quad_.order();

    phi_.assign(nc, 0.0);
    psi_.assign(nd, std::vector<double>(nc, 0.0));
    fission_source_.assign(nc, 0.0);
}

void SNSolver::set_left_bc(double psi)
{
    left_bc_ = psi;
}

void SNSolver::set_right_bc(double psi)
{
    right_bc_ = psi;
}

void SNSolver::set_dsa_omega(double omega)
{
    if (dsa_) dsa_->set_omega(omega);
}

double SNSolver::dsa_omega() const
{
    return dsa_ ? dsa_->omega() : 1.0;
}

void SNSolver::sweep()
{
    int nd = quad_.order();
    int nc = mesh_.n_cells();
    const auto& mu = quad_.mu();

    for (int d = 0; d < nd; ++d) {
        double mu_d = mu[d];
        double abs_mu = std::abs(mu_d);

        if (mu_d > 0.0) {
            double psi_in = left_bc_;

            for (int i = 0; i < nc; ++i) {
                double dx = mesh_.width(i);
                double st = xs_.sigma_t(i);
                double ss = xs_.sigma_s(i);
                double q = xs_.source(i);

                double src = 0.5 * ss * phi_[i] + 0.5 * q;
                if (!fission_source_.empty() && keff_ > 0.0) {
                    src += 0.5 * fission_source_[i] / keff_;
                }

                double psi_c = (2.0 * abs_mu * psi_in + src * dx)
                             / (2.0 * abs_mu + st * dx);
                double psi_out = 2.0 * psi_c - psi_in;

                if (psi_c < 0.0) {
                    psi_c = 0.0;
                    psi_out = psi_in + src * dx / abs_mu;
                    if (psi_out < 0.0) psi_out = 0.0;
                }

                psi_[d][i] = psi_c;
                psi_in = psi_out;
            }
        } else {
            double psi_in = right_bc_;

            for (int i = nc - 1; i >= 0; --i) {
                double dx = mesh_.width(i);
                double st = xs_.sigma_t(i);
                double ss = xs_.sigma_s(i);
                double q = xs_.source(i);

                double src = 0.5 * ss * phi_[i] + 0.5 * q;
                if (!fission_source_.empty() && keff_ > 0.0) {
                    src += 0.5 * fission_source_[i] / keff_;
                }

                double psi_c = (2.0 * abs_mu * psi_in + src * dx)
                             / (2.0 * abs_mu + st * dx);
                double psi_out = 2.0 * psi_c - psi_in;

                if (psi_c < 0.0) {
                    psi_c = 0.0;
                    psi_out = psi_in + src * dx / abs_mu;
                    if (psi_out < 0.0) psi_out = 0.0;
                }

                psi_[d][i] = psi_c;
                psi_in = psi_out;
            }
        }
    }
}

void SNSolver::compute_scalar_flux()
{
    int nd = quad_.order();
    int nc = mesh_.n_cells();
    const auto& wt = quad_.wt();

    std::fill(phi_.begin(), phi_.end(), 0.0);

    for (int d = 0; d < nd; ++d) {
        for (int i = 0; i < nc; ++i) {
            phi_[i] += wt[d] * psi_[d][i];
        }
    }
}

double SNSolver::compute_error(const std::vector<double>& phi_old) const
{
    double max_err = 0.0;
    for (int i = 0; i < mesh_.n_cells(); ++i) {
        double denom = std::abs(phi_[i]) + 1e-30;
        double err = std::abs(phi_[i] - phi_old[i]) / denom;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

void SNSolver::apply_dsa(const std::vector<double>& phi_old)
{
    if (!dsa_enabled_ || !dsa_) return;

    int nc = mesh_.n_cells();
    std::vector<double> residual(nc, 0.0);

    for (int i = 0; i < nc; ++i) {
        residual[i] = phi_[i] - phi_old[i];
    }

    std::vector<double> correction = dsa_->solve(residual);

    for (int i = 0; i < nc; ++i) {
        double phi_new = phi_[i] + correction[i];
        if (phi_new < 0.0) {
            if (phi_[i] > 1e-30) {
                phi_[i] *= 0.5;
            } else {
                phi_[i] = 0.0;
            }
        } else {
            phi_[i] = phi_new;
        }
    }
}

SolverResult SNSolver::solve()
{
    int nc = mesh_.n_cells();
    std::fill(phi_.begin(), phi_.end(), 1.0);

    int iter = 0;
    double err = 1.0;
    bool converged = false;
    double prev_err = 1e30;
    int dsa_fail_count = 0;

    for (iter = 1; iter <= max_iter_; ++iter) {
        std::vector<double> phi_old = phi_;

        sweep();
        compute_scalar_flux();

        apply_dsa(phi_old);

        err = compute_error(phi_old);

        if (dsa_enabled_ && dsa_ && err > prev_err * 1.5 && iter > 3) {
            dsa_fail_count++;
            if (dsa_fail_count >= 3) {
                double omega = dsa_->omega();
                if (omega > 0.1) {
                    dsa_->set_omega(omega * 0.5);
                    dsa_fail_count = 0;
                } else {
                    dsa_enabled_ = false;
                }
            }
        } else {
            dsa_fail_count = 0;
        }

        prev_err = err;

        if (err < tol_) {
            converged = true;
            break;
        }
    }

    return {phi_, std::vector<double>(), iter, err, converged};
}

SolverResult SNSolver::solve_eigenvalue(double initial_keff)
{
    int nc = mesh_.n_cells();
    keff_ = initial_keff;

    std::fill(phi_.begin(), phi_.end(), 1.0);
    fission_source_.assign(nc, 1.0);

    int iter = 0;
    double err = 1.0;
    double keff_err = 1.0;
    bool converged = false;

    for (iter = 1; iter <= max_iter_; ++iter) {
        std::vector<double> phi_old = phi_;
        double keff_old = keff_;

        for (int i = 0; i < nc; ++i) {
            double nu_sigma_f_val = xs_.nu_sigma_f(i);
            fission_source_[i] = nu_sigma_f_val * phi_[i];
        }

        double F_old = std::accumulate(fission_source_.begin(), fission_source_.end(), 0.0);

        sweep();
        compute_scalar_flux();

        apply_dsa(phi_old);

        for (int i = 0; i < nc; ++i) {
            double nu_sigma_f_val = xs_.nu_sigma_f(i);
            fission_source_[i] = nu_sigma_f_val * phi_[i];
        }

        double F_new = std::accumulate(fission_source_.begin(), fission_source_.end(), 0.0);

        if (F_new > 0.0) {
            keff_ = keff_old * F_new / F_old;
        }

        err = compute_error(phi_old);
        keff_err = std::abs(keff_ - keff_old) / (std::abs(keff_) + 1e-30);

        if (err < tol_ && keff_err < tol_) {
            converged = true;
            break;
        }
    }

    return {phi_, std::vector<double>(), iter, err, converged};
}
