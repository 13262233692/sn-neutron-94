#pragma once

#include <vector>
#include <utility>

class GaussLegendreQuadrature {
public:
    explicit GaussLegendreQuadrature(int order);

    int order() const { return order_; }
    const std::vector<double>& mu() const { return mu_; }
    const std::vector<double>& wt() const { return wt_; }

private:
    int order_;
    std::vector<double> mu_;
    std::vector<double> wt_;

    void generate();
    static std::pair<double, double> legendre_root_and_weight(int n, int k);
};
