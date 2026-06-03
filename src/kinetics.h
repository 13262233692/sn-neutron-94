#pragma once

#include <vector>
#include <functional>

struct DelayedNeutronData {
    static constexpr int MAX_GROUPS = 6;
    double beta[MAX_GROUPS];
    double lambda[MAX_GROUPS];
    int n_groups;

    DelayedNeutronData();
    double beta_total() const;
    void set_beta(int i, double val) { beta[i] = val; }
    void set_lambda(int i, double val) { lambda[i] = val; }
    double get_beta(int i) const { return beta[i]; }
    double get_lambda(int i) const { return lambda[i]; }
};

struct KineticsStepResult {
    double time;
    double power;
    std::vector<double> precursor_conc;
    double reactivity;
};

struct KineticsResult {
    std::vector<double> time;
    std::vector<double> power;
    std::vector<std::vector<double>> precursor_conc;
    std::vector<double> reactivity;
    double final_time;
    int n_steps;
};

class PointKinetics {
public:
    using ReactivityFunc = std::function<double(double t)>;

    PointKinetics(const DelayedNeutronData& dn_data,
                  double Lambda,
                  double initial_power = 1.0,
                  double initial_source = 0.0);

    void set_reactivity(double rho);
    void set_source(double S);
    void set_reactivity_function(ReactivityFunc rho_func);

    KineticsResult solve(double t_end, double dt, double theta = 1.0);

    double power() const { return P_; }
    double reactivity() const { return rho_; }
    const std::vector<double>& precursors() const { return C_; }
    const DelayedNeutronData& dn_data() const { return dn_; }
    double generation_time() const { return Lambda_; }

private:
    DelayedNeutronData dn_;
    double Lambda_;
    double P_;
    double rho_;
    double S_;
    std::vector<double> C_;
    ReactivityFunc rho_func_;

    void compute_initial_precursors();
};
