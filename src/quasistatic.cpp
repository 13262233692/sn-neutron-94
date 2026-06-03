#include "quasistatic.h"
#include "solver.h"
#include <cmath>
#include <algorithm>
#include <numeric>

QuasiStaticSolver::QuasiStaticSolver(const GaussLegendreQuadrature& quad,
                                     const Mesh1D& mesh,
                                     const XSData& xs,
                                     const DelayedNeutronData& dn_data,
                                     double Lambda,
                                     double nu_sigma_f_ref,
                                     double tol,
                                     int max_iter)
    : quad_(quad), mesh_(mesh), xs_base_(xs), xs_current_(xs),
      dn_data_(dn_data), Lambda_(Lambda), nu_sigma_f_ref_(nu_sigma_f_ref),
      tol_(tol), max_iter_(max_iter),
      left_bc_(0.0), right_bc_(0.0),
      dsa_enabled_(true), use_eigenvalue_(false),
      keff_(1.0),
      xs_change_func_(nullptr)
{
    int nc = mesh_.n_cells();
    shape_.assign(nc, 1.0);
    phi_.assign(nc, 1.0);
    nu_sigma_f_.assign(nc, nu_sigma_f_ref);
    compute_nu_sigma_f();
}

void QuasiStaticSolver::set_left_bc(double psi) { left_bc_ = psi; }
void QuasiStaticSolver::set_right_bc(double psi) { right_bc_ = psi; }
void QuasiStaticSolver::set_dsa_enabled(bool enabled) { dsa_enabled_ = enabled; }
void QuasiStaticSolver::set_eigenvalue_mode(bool use_eigenvalue) { use_eigenvalue_ = use_eigenvalue; }

void QuasiStaticSolver::set_xs_change_function(XSChangeFunc func)
{
    xs_change_func_ = std::move(func);
}

void QuasiStaticSolver::compute_nu_sigma_f()
{
    int nc = mesh_.n_cells();
    for (int i = 0; i < nc; ++i) {
        nu_sigma_f_[i] = xs_current_.nu_sigma_f(i);
    }
}

void QuasiStaticSolver::normalize_shape()
{
    int nc = mesh_.n_cells();
    double norm = 0.0;
    for (int i = 0; i < nc; ++i) {
        norm += nu_sigma_f_[i] * shape_[i] * mesh_.width(i);
    }
    if (norm > 1e-30) {
        double inv_norm = 1.0 / norm;
        for (int i = 0; i < nc; ++i) {
            shape_[i] *= inv_norm;
        }
    }
}

void QuasiStaticSolver::compute_kinetics_params(double& rho, double& beta_eff, double& gen_time)
{
    double bt = dn_data_.beta_total();

    if (keff_ < 1e-30) keff_ = 1e-30;
    rho = (keff_ - 1.0) / keff_;

    beta_eff = bt;

    gen_time = Lambda_;
}

void QuasiStaticSolver::update_shape()
{
    int nc = mesh_.n_cells();

    SNSolver solver(quad_, mesh_, xs_current_, tol_, max_iter_);
    solver.set_left_bc(left_bc_);
    solver.set_right_bc(right_bc_);
    solver.set_dsa_enabled(dsa_enabled_);

    if (use_eigenvalue_) {
        SolverResult result = solver.solve_eigenvalue(keff_);
        keff_ = solver.keff();
        for (int i = 0; i < nc; ++i) {
            shape_[i] = result.scalar_flux[i];
        }
    } else {
        SolverResult result = solver.solve();
        for (int i = 0; i < nc; ++i) {
            shape_[i] = result.scalar_flux[i];
        }
    }

    normalize_shape();
}

void QuasiStaticSolver::apply_xs_changes(const std::vector<XSChange>& changes)
{
    for (const auto& ch : changes) {
        xs_current_.set_cell_xs(ch.cell, ch.sigma_t, ch.sigma_s, ch.nu_sigma_f, ch.source);
    }
    compute_nu_sigma_f();
}

QuasiStaticResult QuasiStaticSolver::solve(double t_end,
                                           double dt_shape,
                                           double dt_kinetics,
                                           double initial_power,
                                           double theta_kinetics)
{
    QuasiStaticResult result;
    int nc = mesh_.n_cells();

    compute_initial_shape();

    double rho0, beta_eff0, gen_time0;
    compute_kinetics_params(rho0, beta_eff0, gen_time0);

    double T = initial_power;
    for (int i = 0; i < nc; ++i) {
        phi_[i] = T * shape_[i];
    }

    result.time.push_back(0.0);
    result.amplitude.push_back(T);
    result.reactivity.push_back(rho0);
    result.beta_eff.push_back(beta_eff0);
    result.gen_time.push_back(gen_time0);
    result.shape_flux.push_back(shape_);
    result.full_flux.push_back(phi_);

    int n_shape_updates = 0;
    int total_kinetics_steps = 0;

    PointKinetics pk(dn_data_, Lambda_, T, 0.0);
    pk.set_reactivity(rho0);

    double t = 0.0;
    double t_next_shape = dt_shape;

    while (t < t_end - 1e-12) {
        double dt = std::min(dt_kinetics, t_end - t);

        if (t + dt > t_next_shape + 1e-12 && t_next_shape <= t_end) {
            dt = std::min(dt, t_next_shape - t);
        }

        bool do_shape_update = false;
        if (t + dt >= t_next_shape - 1e-12 && t < t_next_shape - 1e-12) {
            do_shape_update = true;
        }

        double rho_now, beta_eff_now, gen_time_now;
        compute_kinetics_params(rho_now, beta_eff_now, gen_time_now);

        pk.set_reactivity(rho_now);

        double dt_pk = std::min(dt, t_end - t);
        double pk_dt_internal = std::min(dt_pk * 0.1, 1e-5);
        if (pk_dt_internal < 1e-8) pk_dt_internal = dt_pk;

        KineticsResult pk_result = pk.solve(dt_pk, pk_dt_internal, theta_kinetics);

        T = pk_result.power.back();
        total_kinetics_steps += pk_result.n_steps;

        t += dt_pk;

        for (int i = 0; i < nc; ++i) {
            phi_[i] = T * shape_[i];
        }

        if (do_shape_update) {
            if (xs_change_func_) {
                auto changes = xs_change_func_(t);
                apply_xs_changes(changes);
            }
            update_shape();
            n_shape_updates++;

            double rho_new, beta_eff_new, gen_time_new;
            compute_kinetics_params(rho_new, beta_eff_new, gen_time_new);
            pk.set_reactivity(rho_new);

            for (int i = 0; i < nc; ++i) {
                phi_[i] = T * shape_[i];
            }

            t_next_shape += dt_shape;
        }

        result.time.push_back(t);
        result.amplitude.push_back(T);
        result.reactivity.push_back(rho_now);
        result.beta_eff.push_back(beta_eff_now);
        result.gen_time.push_back(gen_time_now);
        result.shape_flux.push_back(shape_);
        result.full_flux.push_back(phi_);
    }

    result.initial_keff = keff_;
    result.n_shape_updates = n_shape_updates;
    result.total_kinetics_steps = total_kinetics_steps;

    return result;
}

void QuasiStaticSolver::compute_initial_shape()
{
    int nc = mesh_.n_cells();

    SNSolver solver(quad_, mesh_, xs_current_, tol_, max_iter_);
    solver.set_left_bc(left_bc_);
    solver.set_right_bc(right_bc_);
    solver.set_dsa_enabled(dsa_enabled_);

    if (use_eigenvalue_) {
        SolverResult result = solver.solve_eigenvalue(1.0);
        keff_ = solver.keff();
        for (int i = 0; i < nc; ++i) {
            shape_[i] = result.scalar_flux[i];
        }
    } else {
        SolverResult result = solver.solve();
        for (int i = 0; i < nc; ++i) {
            shape_[i] = result.scalar_flux[i];
        }
    }

    normalize_shape();
}
