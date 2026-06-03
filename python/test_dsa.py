import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from snsolver import solve_slab, Quadrature, Mesh1D, XSData, SNSolver


def test_dsa_vs_no_dsa_high_scattering():
    print("=" * 70)
    print("Test 1: DSA vs No-DSA — High scattering ratio (c=0.99)")
    print("=" * 70)

    sn_order = 8
    n_cells = 200
    sigma_t = 1.0
    sigma_s = 0.99
    source = 1.0

    result_dsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                            sigma_t, sigma_s, source,
                            tol=1e-8, max_iter=10000, dsa=True)

    result_nodsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                              sigma_t, sigma_s, source,
                              tol=1e-8, max_iter=10000, dsa=False)

    phi_dsa = result_dsa['scalar_flux']
    phi_nodsa = result_nodsa['scalar_flux']
    centers = result_dsa['centers']

    flux_diff = np.max(np.abs(phi_dsa - phi_nodsa) / (np.abs(phi_dsa) + 1e-30))
    speedup = result_nodsa['iterations'] / max(result_dsa['iterations'], 1)

    print("  c = sigma_s/sigma_t = %.2f" % (sigma_s / sigma_t))
    print("  DSA:   %5d iters, converged=%s, err=%.2e" % (result_dsa['iterations'], result_dsa['converged'], result_dsa['error']))
    print("  NoDSA: %5d iters, converged=%s, err=%.2e" % (result_nodsa['iterations'], result_nodsa['converged'], result_nodsa['error']))
    print("  Speedup: %.1fx" % speedup)
    print("  Max flux diff DSA vs NoDSA: %.2e" % flux_diff)

    assert result_dsa['converged'], "DSA solver did not converge!"
    assert result_dsa['iterations'] < result_nodsa['iterations'], \
        "DSA should be faster: %d vs %d" % (result_dsa['iterations'], result_nodsa['iterations'])
    assert flux_diff < 0.01, "Fluxes differ by %.2e" % flux_diff

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, phi_dsa, 'b-', linewidth=2, label='DSA (%d iters)' % result_dsa['iterations'])
    ax.plot(centers, phi_nodsa, 'r--', linewidth=1.5, label='No DSA (%d iters)' % result_nodsa['iterations'])
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Scalar Flux')
    ax.set_title('Scalar Flux: c=0.99 (S%d)' % sn_order)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test_dsa_high_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test_dsa_high_scatter.png")


def test_dsa_strong_absorber():
    print("\n" + "=" * 70)
    print("Test 2: Strong absorber with scattering (heterogeneous)")
    print("=" * 70)

    sn_order = 8
    n_cells = 200

    sigma_t = np.zeros(n_cells)
    sigma_s = np.zeros(n_cells)
    source_arr = np.zeros(n_cells)

    n_source = n_cells // 3
    n_absorb = n_cells // 3

    sigma_t[:n_source] = 1.0
    sigma_s[:n_source] = 0.8
    source_arr[:n_source] = 10.0

    sigma_t[n_source:n_source + n_absorb] = 10.0
    sigma_s[n_source:n_source + n_absorb] = 5.0
    source_arr[n_source:n_source + n_absorb] = 0.0

    sigma_t[n_source + n_absorb:] = 1.0
    sigma_s[n_source + n_absorb:] = 0.5
    source_arr[n_source + n_absorb:] = 0.0

    result_dsa = solve_slab(sn_order, 0.0, 20.0, n_cells,
                            sigma_t, sigma_s, source_arr,
                            tol=1e-8, max_iter=10000, dsa=True)

    result_nodsa = solve_slab(sn_order, 0.0, 20.0, n_cells,
                              sigma_t, sigma_s, source_arr,
                              tol=1e-8, max_iter=10000, dsa=False)

    phi_dsa = result_dsa['scalar_flux']
    phi_nodsa = result_nodsa['scalar_flux']
    centers = result_dsa['centers']

    flux_diff = np.max(np.abs(phi_dsa - phi_nodsa) / (np.abs(phi_nodsa) + 1e-30))

    print("  Source region:   [0, %.2f] cm (c=0.8)" % centers[n_source - 1])
    print("  Absorber region: [%.2f, %.2f] cm (c=0.5, St=10)" % (centers[n_source], centers[n_source + n_absorb - 1]))
    print("  Shield region:   [%.2f, 20.0] cm (c=0.5)" % centers[n_source + n_absorb])
    print("  DSA:   %5d iters, converged=%s, err=%.2e" % (result_dsa['iterations'], result_dsa['converged'], result_dsa['error']))
    print("  NoDSA: %5d iters, converged=%s, err=%.2e" % (result_nodsa['iterations'], result_nodsa['converged'], result_nodsa['error']))
    print("  Flux match: %.2e" % flux_diff)

    assert result_dsa['converged'], "DSA solver did not converge for strong absorber!"
    assert flux_diff < 0.01, "Fluxes differ by %.2e" % flux_diff

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(centers, np.maximum(phi_dsa, 1e-30), 'b-', linewidth=2, label='DSA (%d iters)' % result_dsa['iterations'])
    ax.semilogy(centers, np.maximum(phi_nodsa, 1e-30), 'r--', linewidth=1.5, label='No DSA (%d iters)' % result_nodsa['iterations'])
    ax.axvline(x=centers[n_source - 1], color='k', linestyle='--', alpha=0.5, label='Source/Absorber')
    ax.axvline(x=centers[n_source + n_absorb - 1], color='g', linestyle='--', alpha=0.5, label='Absorber/Shield')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Scalar Flux')
    ax.set_title('Strong Absorber Problem: DSA vs No-DSA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test_dsa_strong_absorber.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test_dsa_strong_absorber.png")


def test_dsa_speedup_table():
    print("\n" + "=" * 70)
    print("Test 3: DSA Speedup vs Scattering Ratio c")
    print("=" * 70)

    sn_order = 8
    n_cells = 100
    sigma_t = 1.0
    source = 1.0

    c_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    results = []

    print("  %8s  %10s  %12s  %8s  %10s" % ("c", "DSA iters", "NoDSA iters", "Speedup", "Flux match"))
    print("  %8s  %10s  %12s  %8s  %10s" % ("-" * 8, "-" * 10, "-" * 12, "-" * 8, "-" * 10))

    for c in c_values:
        sigma_s = c * sigma_t

        result_dsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                                sigma_t, sigma_s, source,
                                tol=1e-8, max_iter=10000, dsa=True)

        result_nodsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                                  sigma_t, sigma_s, source,
                                  tol=1e-8, max_iter=10000, dsa=False)

        phi_dsa = result_dsa['scalar_flux']
        phi_nodsa = result_nodsa['scalar_flux']
        max_diff = np.max(np.abs(phi_dsa - phi_nodsa) / (np.abs(phi_dsa) + 1e-30))

        dsa_iters = result_dsa['iterations']
        nodsa_iters = result_nodsa['iterations']
        speedup = nodsa_iters / max(dsa_iters, 1)

        results.append((c, dsa_iters, nodsa_iters, speedup, max_diff))

        print("  %8.3f  %10d  %12d  %7.1fx  %10.2e" % (c, dsa_iters, nodsa_iters, speedup, max_diff))

    fig, ax = plt.subplots(figsize=(10, 6))
    c_arr = [r[0] for r in results]
    speedup_arr = [r[3] for r in results]
    ax.semilogy(c_arr, speedup_arr, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Scattering ratio c = Sigma_s / Sigma_t')
    ax.set_ylabel('DSA Speedup (iterations ratio)')
    ax.set_title('DSA Acceleration Effect vs Scattering Ratio (S%d)' % sn_order)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test_dsa_speedup.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test_dsa_speedup.png")

    for c_val, dsa_it, nodsa_it, sp, diff in results:
        if c_val >= 0.9:
            assert dsa_it < nodsa_it, "DSA should be faster for c=%.2f: %d vs %d" % (c_val, dsa_it, nodsa_it)


def test_dsa_flux_consistency():
    print("\n" + "=" * 70)
    print("Test 4: DSA flux consistency with very tight tolerance")
    print("=" * 70)

    sn_order = 16
    n_cells = 200
    sigma_t = 1.0
    sigma_s = 0.99
    source = 1.0

    result_dsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                            sigma_t, sigma_s, source,
                            tol=1e-12, max_iter=20000, dsa=True)

    result_nodsa = solve_slab(sn_order, 0.0, 10.0, n_cells,
                              sigma_t, sigma_s, source,
                              tol=1e-12, max_iter=20000, dsa=False)

    phi_dsa = result_dsa['scalar_flux']
    phi_nodsa = result_nodsa['scalar_flux']

    rel_diff = np.max(np.abs(phi_dsa - phi_nodsa) / (np.abs(phi_dsa) + 1e-30))

    print("  DSA:   %d iters, converged=%s" % (result_dsa['iterations'], result_dsa['converged']))
    print("  NoDSA: %d iters, converged=%s" % (result_nodsa['iterations'], result_nodsa['converged']))
    print("  Max relative flux difference: %.2e" % rel_diff)

    assert result_dsa['converged'], "DSA did not converge with tight tolerance!"
    if result_nodsa['converged']:
        assert rel_diff < 1e-4, "Flux solutions differ by %.2e" % rel_diff

    print("  PASSED: DSA and NoDSA produce consistent results")


def test_dsa_low_level_api():
    print("\n" + "=" * 70)
    print("Test 5: Low-level DSA API (omega control)")
    print("=" * 70)

    quad = Quadrature(8)
    mesh = Mesh1D(0.0, 10.0, 100)
    xs = XSData()
    for i in range(100):
        xs.set_cell_xs(i, 1.0, 0.95, 1.0)

    solver = SNSolver(quad, mesh, xs, 1e-8, 1000)
    solver.set_left_bc(0.0)
    solver.set_right_bc(0.0)

    print("  Default DSA enabled: %s" % solver.dsa_enabled)
    print("  Default DSA omega: %.1f" % solver.dsa_omega)

    assert solver.dsa_enabled, "DSA should be enabled by default"
    assert abs(solver.dsa_omega - 1.0) < 1e-10, "Default omega should be 1.0"

    result_dsa = solver.solve()
    iters_dsa = result_dsa.iterations

    solver2 = SNSolver(quad, mesh, xs, 1e-8, 1000)
    solver2.set_left_bc(0.0)
    solver2.set_right_bc(0.0)
    solver2.set_dsa_enabled(False)
    result_nodsa = solver2.solve()
    iters_nodsa = result_nodsa.iterations

    print("  DSA (omega=1.0): %d iters" % iters_dsa)
    print("  No DSA: %d iters" % iters_nodsa)
    print("  Speedup: %.1fx" % (iters_nodsa / max(iters_dsa, 1)))

    assert result_dsa.converged, "DSA did not converge!"
    assert iters_dsa < iters_nodsa, "DSA should be faster"

    print("  PASSED: Low-level DSA API works correctly")


if __name__ == '__main__':
    test_dsa_vs_no_dsa_high_scattering()
    test_dsa_strong_absorber()
    test_dsa_speedup_table()
    test_dsa_flux_consistency()
    test_dsa_low_level_api()
    print("\n" + "=" * 70)
    print("All DSA tests completed!")
    print("=" * 70)
