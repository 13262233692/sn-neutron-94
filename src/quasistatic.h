#pragma once

#include "quadrature.h"
#include "mesh.h"
#include "xsdata.h"
#include "kinetics.h"
#include <vector>
#include <functional>
#include <memory>

struct XSChange {
    int cell = 0;
    double sigma_t = 0.0;
    double sigma_s = 0.0;
    double nu_sigma_f = 0.0;
    double source = 0.0;

    XSChange() = default;
    XSChange(int c, double st, double ss, double nsf, double src)
        : cell(c), sigma_t(st), sigma_s(ss), nu_sigma_f(nsf), source(src) {}
};

struct QuasiStaticResult {
    std::vector<double> time;
    std::vector<double> amplitude;
    std::vector<double> reactivity;
    std::vector<double> beta_eff;
    std::vector<double> gen_time;
    std::vector<std::vector<double>> shape_flux;
    std::vector<std::vector<double>> full_flux;
    double initial_keff;
    int n_shape_updates;
    int total_kinetics_steps;
};

class QuasiStaticSolver {
public:
    using XSChangeFunc = std::function<std::vector<XSChange>(double t)>;

    QuasiStaticSolver(const GaussLegendreQuadrature& quad,
                      const Mesh1D& mesh,
                      const XSData& xs,
                      const DelayedNeutronData& dn_data,
                      double Lambda,
                      double nu_sigma_f_ref,
                      double tol = 1e-6,
                      int max_iter = 1000);

    void set_left_bc(double psi);
    void set_right_bc(double psi);
    void set_xs_change_function(XSChangeFunc func);
    void set_dsa_enabled(bool enabled);
    void set_eigenvalue_mode(bool use_eigenvalue);

    QuasiStaticResult solve(double t_end,
                            double dt_shape,
                            double dt_kinetics,
                            double initial_power = 1.0,
                            double theta_kinetics = 1.0);

    const std::vector<double>& shape() const { return shape_; }

private:
    GaussLegendreQuadrature quad_;
    Mesh1D mesh_;
    XSData xs_base_;
    XSData xs_current_;
    DelayedNeutronData dn_data_;
    double Lambda_;
    double nu_sigma_f_ref_;
    double tol_;
    int max_iter_;
    double left_bc_;
    double right_bc_;
    bool dsa_enabled_;
    bool use_eigenvalue_;

    std::vector<double> shape_;
    std::vector<double> phi_;
    std::vector<double> nu_sigma_f_;
    double keff_;

    XSChangeFunc xs_change_func_;

    void compute_initial_shape();
    void compute_kinetics_params(double& rho, double& beta_eff, double& gen_time);
    void update_shape();
    void normalize_shape();
    void apply_xs_changes(const std::vector<XSChange>& changes);
    void compute_nu_sigma_f();
};
