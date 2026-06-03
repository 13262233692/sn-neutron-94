#pragma once

#include "quadrature.h"
#include "mesh.h"
#include "xsdata.h"
#include "dsa.h"
#include <vector>
#include <memory>

struct SolverResult {
    std::vector<double> scalar_flux;
    std::vector<double> scalar_flux_old;
    int iterations;
    double error;
    bool converged;
};

class SNSolver {
public:
    SNSolver(const GaussLegendreQuadrature& quad,
             const Mesh1D& mesh,
             const XSData& xs,
             double tol = 1e-6,
             int max_iter = 10000);

    SolverResult solve();

    void set_left_bc(double psi);
    void set_right_bc(double psi);

    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_dsa_enabled(bool enabled) { dsa_enabled_ = enabled; }
    void set_dsa_omega(double omega);

    bool dsa_enabled() const { return dsa_enabled_; }
    double dsa_omega() const;

    const std::vector<double>& scalar_flux() const { return phi_; }

    const std::vector<std::vector<double>>& angular_flux() const { return psi_; }

    double keff() const { return keff_; }

    SolverResult solve_eigenvalue(double initial_keff = 1.0);

private:
    GaussLegendreQuadrature quad_;
    Mesh1D mesh_;
    XSData xs_;

    double tol_;
    int max_iter_;

    double left_bc_;
    double right_bc_;

    std::vector<double> phi_;
    std::vector<std::vector<double>> psi_;

    double keff_;
    std::vector<double> fission_source_;

    bool dsa_enabled_;
    std::unique_ptr<DSA> dsa_;

    void sweep();
    void compute_scalar_flux();
    double compute_error(const std::vector<double>& phi_old) const;
    void apply_dsa(const std::vector<double>& phi_old);
};
