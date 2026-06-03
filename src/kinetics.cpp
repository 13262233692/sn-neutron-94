#include "kinetics.h"
#include <cmath>
#include <algorithm>

DelayedNeutronData::DelayedNeutronData()
    : n_groups(6)
{
    beta[0] = 0.000215;
    beta[1] = 0.001424;
    beta[2] = 0.001274;
    beta[3] = 0.002568;
    beta[4] = 0.000748;
    beta[5] = 0.000273;

    lambda[0] = 0.0124;
    lambda[1] = 0.0305;
    lambda[2] = 0.111;
    lambda[3] = 0.301;
    lambda[4] = 1.14;
    lambda[5] = 3.01;
}

double DelayedNeutronData::beta_total() const
{
    double s = 0.0;
    for (int i = 0; i < n_groups; ++i) s += beta[i];
    return s;
}

PointKinetics::PointKinetics(const DelayedNeutronData& dn_data,
                             double Lambda,
                             double initial_power,
                             double initial_source)
    : dn_(dn_data), Lambda_(Lambda), P_(initial_power),
      rho_(0.0), S_(initial_source),
      C_(dn_data.n_groups, 0.0),
      rho_func_(nullptr)
{
    compute_initial_precursors();
}

void PointKinetics::compute_initial_precursors()
{
    for (int i = 0; i < dn_.n_groups; ++i) {
        C_[i] = dn_.beta[i] * P_ / (Lambda_ * dn_.lambda[i]);
    }
}

void PointKinetics::set_reactivity(double rho)
{
    rho_ = rho;
}

void PointKinetics::set_source(double S)
{
    S_ = S;
}

void PointKinetics::set_reactivity_function(ReactivityFunc rho_func)
{
    rho_func_ = std::move(rho_func);
}

KineticsResult PointKinetics::solve(double t_end, double dt, double theta)
{
    KineticsResult result;
    int n_steps = static_cast<int>(std::ceil(t_end / dt));
    double t = 0.0;
    double bt = dn_.beta_total();

    result.time.push_back(t);
    result.power.push_back(P_);
    result.precursor_conc.push_back(C_);
    result.reactivity.push_back(rho_func_ ? rho_func_(t) : rho_);

    for (int step = 0; step < n_steps; ++step) {
        double actual_dt = std::min(dt, t_end - t);

        double rho_n = rho_func_ ? rho_func_(t) : rho_;
        double rho_np1 = rho_func_ ? rho_func_(t + actual_dt) : rho_;
        double rho_bar = theta * rho_np1 + (1.0 - theta) * rho_n;

        double rhs = P_;
        double lhs_coeff = 1.0;

        for (int i = 0; i < dn_.n_groups; ++i) {
            rhs += actual_dt * (1.0 - theta) * dn_.lambda[i] * C_[i];
            rhs += actual_dt * theta * dn_.lambda[i] * C_[i] /
                   (1.0 + theta * actual_dt * dn_.lambda[i]);

            double gi_coeff = theta * actual_dt * dn_.beta[i] * dn_.lambda[i] /
                              (Lambda_ * (1.0 + theta * actual_dt * dn_.lambda[i]));
            lhs_coeff -= theta * actual_dt * gi_coeff;
        }

        rhs += actual_dt * (1.0 - theta) * ((rho_n - bt) / Lambda_ * P_ + S_);
        rhs += actual_dt * theta * S_;

        lhs_coeff -= theta * actual_dt * (rho_bar - bt) / Lambda_;

        if (std::abs(lhs_coeff) < 1e-30) lhs_coeff = 1e-30;

        double P_new = rhs / lhs_coeff;
        if (P_new < 0.0) P_new = 0.0;

        for (int i = 0; i < dn_.n_groups; ++i) {
            double C_new = (C_[i] + actual_dt * (1.0 - theta) *
                          (dn_.beta[i] * P_ / Lambda_ - dn_.lambda[i] * C_[i]) +
                          theta * actual_dt * dn_.beta[i] * P_new / Lambda_) /
                         (1.0 + theta * actual_dt * dn_.lambda[i]);
            if (C_new < 0.0) C_new = 0.0;
            C_[i] = C_new;
        }

        P_ = P_new;
        rho_ = rho_np1;
        t += actual_dt;

        result.time.push_back(t);
        result.power.push_back(P_);
        result.precursor_conc.push_back(C_);
        result.reactivity.push_back(rho_func_ ? rho_func_(t) : rho_);
    }

    result.final_time = t;
    result.n_steps = n_steps;

    return result;
}
