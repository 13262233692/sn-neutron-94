#define _USE_MATH_DEFINES
#include "quadrature.h"
#include <cmath>

GaussLegendreQuadrature::GaussLegendreQuadrature(int order)
    : order_(order), mu_(order), wt_(order)
{
    generate();
}

static void legendre_eval(int n, double x, double& pn, double& pnm1)
{
    double p0 = 1.0;
    double p1 = x;
    for (int i = 2; i <= n; ++i) {
        double p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / static_cast<double>(i);
        p0 = p1;
        p1 = p2;
    }
    if (n == 0) {
        pn = 1.0;
        pnm1 = 0.0;
    } else if (n == 1) {
        pn = p1;
        pnm1 = p0;
    } else {
        pn = p1;
        pnm1 = p0;
    }
}

void GaussLegendreQuadrature::generate()
{
    int n = order_;
    for (int k = 0; k < n; ++k) {
        auto [root, weight] = legendre_root_and_weight(n, k);
        mu_[k] = root;
        wt_[k] = weight;
    }
}

std::pair<double, double> GaussLegendreQuadrature::legendre_root_and_weight(int n, int k)
{
    double x = std::cos(M_PI * (k + 0.75) / (n + 0.5));

    constexpr int max_iter = 100;
    constexpr double tol = 1e-15;

    for (int iter = 0; iter < max_iter; ++iter) {
        double pn, pnm1;
        legendre_eval(n, x, pn, pnm1);

        double dp = n * (x * pn - pnm1) / (x * x - 1.0);
        double dx = -pn / dp;
        x += dx;

        if (std::abs(dx) < tol) break;
    }

    double pn, pnm1;
    legendre_eval(n, x, pn, pnm1);

    double dp = n * (x * pn - pnm1) / (x * x - 1.0);
    double w = 2.0 / ((1.0 - x * x) * dp * dp);

    return {x, w};
}
