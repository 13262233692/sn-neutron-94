import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from snsolver import solve_slab, Quadrature, Mesh1D, XSData, SNSolver


def test_isotropic_source():
    print("=" * 60)
    print("Test 1: Isotropic source in slab")
    print("=" * 60)

    sn_order = 8
    x_min, x_max = 0.0, 10.0
    n_cells = 100
    sigma_t = 1.0
    sigma_s = 0.9
    source = 1.0

    result = solve_slab(sn_order, x_min, x_max, n_cells,
                        sigma_t, sigma_s, source,
                        left_bc=0.0, right_bc=0.0,
                        tol=1e-8, max_iter=10000)

    phi = result['scalar_flux']
    centers = result['centers']

    phi_inf = source / (sigma_t - sigma_s)
    mid_flux = phi[n_cells // 2]

    print(f"  SN order:        S{sn_order}")
    print(f"  Cells:           {n_cells}")
    print(f"  Sigma_t:         {sigma_t}")
    print(f"  Sigma_s:         {sigma_s}")
    print(f"  Source:          {source}")
    print(f"  c (scat ratio):  {sigma_s/sigma_t}")
    print(f"  Iterations:      {result['iterations']}")
    print(f"  Converged:       {result['converged']}")
    print(f"  Final error:     {result['error']:.2e}")
    print(f"  Midpoint flux:   {mid_flux:.8f}")
    print(f"  Avg scalar flux: {np.mean(phi):.8f}")
    print(f"  Inf. medium:     {phi_inf:.8f}")
    print(f"  Mid/Inf ratio:   {mid_flux/phi_inf:.6f}")

    assert result['converged'], "Source iteration did not converge!"
    assert mid_flux > 0, "Midpoint flux should be positive"
    assert mid_flux < phi_inf, "Midpoint flux should be less than infinite medium"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, phi, 'b-', linewidth=2, label=f'S{sn_order} Numerical')
    ax.axhline(y=phi_inf, color='r', linestyle='--', linewidth=1.5,
               label=f'Infinite medium ({phi_inf:.2f})')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Scalar Flux')
    ax.set_title(f'Test 1: Isotropic Source in Slab (c={sigma_s/sigma_t}, {n_cells} cells)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test1_isotropic_source.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test1_isotropic_source.png")


def test_pure_absorber():
    print("\n" + "=" * 60)
    print("Test 2: Pure absorber with isotropic incident flux")
    print("=" * 60)

    sn_order = 32
    x_min, x_max = 0.0, 5.0
    n_cells = 500
    sigma_t = 2.0
    sigma_s = 0.0
    source = 0.0

    result = solve_slab(sn_order, x_min, x_max, n_cells,
                        sigma_t, sigma_s, source,
                        left_bc=1.0, right_bc=0.0,
                        tol=1e-10, max_iter=10000)

    phi = result['scalar_flux']
    centers = result['centers']

    phi_exact = expn(2, sigma_t * centers)

    mask = phi_exact > 1e-15
    l2_error = np.sqrt(np.mean((phi[mask] - phi_exact[mask]) ** 2)) / np.sqrt(np.mean(phi_exact[mask] ** 2))

    print(f"  SN order:        S{sn_order}")
    print(f"  Cells:           {n_cells}")
    print(f"  Sigma_t:         {sigma_t}")
    print(f"  Left BC:         1.0 (isotropic incident)")
    print(f"  Iterations:      {result['iterations']}")
    print(f"  Converged:       {result['converged']}")
    print(f"  L2 rel. error:   {l2_error:.2e}")
    print(f"  Flux at x=0:     {phi[0]:.8f} (exact: {phi_exact[0]:.8f})")
    print(f"  Flux at x=1:     {phi[n_cells//5]:.8f} (exact: {phi_exact[n_cells//5]:.8f})")

    assert result['converged'], "Solver did not converge!"
    assert l2_error < 0.1, f"L2 error too large: {l2_error}"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(centers[mask], phi[mask], 'b-', linewidth=2, label=f'S{sn_order} Numerical')
    ax.semilogy(centers[mask], phi_exact[mask], 'r--', linewidth=1.5, label='E₂ Analytical')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Scalar Flux')
    ax.set_title(f'Test 2: Pure Absorber with Isotropic Incident Flux (S{sn_order})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test2_pure_absorber.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test2_pure_absorber.png")


def test_two_region():
    print("\n" + "=" * 60)
    print("Test 3: Two-region slab (source/shield)")
    print("=" * 60)

    sn_order = 8
    x_min, x_max = 0.0, 20.0
    n_cells = 200

    sigma_t = np.zeros(n_cells)
    sigma_s = np.zeros(n_cells)
    source_arr = np.zeros(n_cells)

    n_source = n_cells // 2

    sigma_t[:n_source] = 1.0
    sigma_s[:n_source] = 0.8
    source_arr[:n_source] = 10.0

    sigma_t[n_source:] = 2.0
    sigma_s[n_source:] = 0.5
    source_arr[n_source:] = 0.0

    result = solve_slab(sn_order, x_min, x_max, n_cells,
                        sigma_t, sigma_s, source_arr,
                        left_bc=0.0, right_bc=0.0,
                        tol=1e-8, max_iter=10000)

    phi = result['scalar_flux']
    centers = result['centers']

    print(f"  SN order:        S{sn_order}")
    print(f"  Cells:           {n_cells}")
    print(f"  Source region:   [{x_min:.0f}, {centers[n_source-1]:.2f}] cm")
    print(f"  Shield region:   [{centers[n_source]:.2f}, {x_max:.0f}] cm")
    print(f"  Iterations:      {result['iterations']}")
    print(f"  Converged:       {result['converged']}")
    print(f"  Max flux:        {np.max(phi):.6f}")
    print(f"  Min flux:        {np.min(phi):.6f}")
    print(f"  Flux at shield start: {phi[n_source]:.6f}")
    print(f"  Flux at shield end:   {phi[-1]:.6e}")

    assert result['converged'], "Solver did not converge!"
    assert np.max(phi) > 0, "Flux should be positive"
    assert phi[n_source] > phi[-1], "Flux should decrease in shield"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(centers, phi, 'b-', linewidth=2)
    ax1.axvline(x=centers[n_source - 1], color='k', linestyle='--', alpha=0.5,
                label='Source/Shield boundary')
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Scalar Flux')
    ax1.set_title(f'Test 3: Two-Region Slab - Linear Scale (S{sn_order})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(centers, np.maximum(phi, 1e-30), 'b-', linewidth=2)
    ax2.axvline(x=centers[n_source - 1], color='k', linestyle='--', alpha=0.5,
                label='Source/Shield boundary')
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Scalar Flux')
    ax2.set_title('Two-Region Slab - Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(os.path.dirname(__file__), 'test3_two_region.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test3_two_region.png")


def test_spatial_convergence():
    print("\n" + "=" * 60)
    print("Test 4: Spatial convergence (mesh refinement)")
    print("=" * 60)

    sn_order = 16
    sigma_t = 1.0
    sigma_s = 0.5
    source = 1.0

    ref_cells = 2000
    ref_result = solve_slab(sn_order, 0.0, 10.0, ref_cells,
                            sigma_t, sigma_s, source,
                            tol=1e-10, max_iter=20000)
    ref_flux = ref_result['scalar_flux']
    ref_centers = ref_result['centers']

    cell_sizes = [10, 20, 50, 100, 200, 500]
    errors = []

    for nc in cell_sizes:
        result = solve_slab(sn_order, 0.0, 10.0, nc,
                            sigma_t, sigma_s, source,
                            tol=1e-10, max_iter=20000)
        phi = result['scalar_flux']
        centers = result['centers']

        ref_at_centers = np.interp(centers, ref_centers, ref_flux)
        l2_err = np.sqrt(np.mean((phi - ref_at_centers) ** 2)) / (np.sqrt(np.mean(ref_at_centers ** 2)) + 1e-30)
        errors.append(l2_err)
        print(f"  Cells={nc:4d}, dx={10.0/nc:.4f}, L2 err={l2_err:.6e}")

    dx_vals = [10.0 / nc for nc in cell_sizes]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(dx_vals, errors, 'bo-', linewidth=2, markersize=8, label='DD Error')
    ax.loglog(dx_vals, [errors[0] * (dx_vals[i] / dx_vals[0]) ** 2 for i in range(len(dx_vals))],
              'r--', linewidth=1, label='O(Δx²) reference')
    ax.set_xlabel('Cell size Δx (cm)')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Spatial Convergence (Diamond Difference)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test4_spatial_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test4_spatial_convergence.png")


def test_quadrature_convergence():
    print("\n" + "=" * 60)
    print("Test 5: Angular convergence (SN order refinement)")
    print("=" * 60)

    n_cells = 200
    sigma_t = 1.0
    sigma_s = 0.9
    source = 1.0

    ref_order = 32
    ref_result = solve_slab(ref_order, 0.0, 10.0, n_cells,
                            sigma_t, sigma_s, source,
                            tol=1e-10, max_iter=20000)
    ref_flux = ref_result['scalar_flux']

    sn_orders = [2, 4, 8, 16]
    errors = []

    for sn in sn_orders:
        result = solve_slab(sn, 0.0, 10.0, n_cells,
                            sigma_t, sigma_s, source,
                            tol=1e-10, max_iter=20000)
        phi = result['scalar_flux']
        l2_err = np.sqrt(np.mean((phi - ref_flux) ** 2)) / (np.sqrt(np.mean(ref_flux ** 2)) + 1e-30)
        errors.append(l2_err)
        print(f"  S{sn:2d}: L2 err={l2_err:.6e}, iters={result['iterations']}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(sn_orders, errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('SN Order')
    ax.set_ylabel('L2 Relative Error vs S32')
    ax.set_title('Angular Convergence (SN Order)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'test5_angular_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Plot saved: test5_angular_convergence.png")


def test_low_level_api():
    print("\n" + "=" * 60)
    print("Test 6: Low-level C++ API")
    print("=" * 60)

    quad = Quadrature(4)
    mesh = Mesh1D(0.0, 5.0, 50)
    xs = XSData()

    for i in range(50):
        xs.set_cell_xs(i, 1.0, 0.5, 2.0)

    solver = SNSolver(quad, mesh, xs, 1e-8, 1000)
    solver.set_left_bc(0.0)
    solver.set_right_bc(0.0)

    result = solver.solve()

    phi = np.array(result.scalar_flux)
    centers = np.array(mesh.centers)
    angular = np.array(solver.angular_flux)

    print(f"  Quadrature order: {quad.order}")
    print(f"  Mu values:        {[f'{m:.6f}' for m in quad.mu]}")
    print(f"  Weights:          {[f'{w:.6f}' for w in quad.wt]}")
    print(f"  Weight sum:       {sum(quad.wt):.10f}")
    print(f"  Mesh cells:       {mesh.n_cells}")
    print(f"  Iterations:       {result.iterations}")
    print(f"  Converged:        {result.converged}")
    print(f"  Angular flux shape: {angular.shape}")
    print(f"  Scalar flux range:  [{np.min(phi):.6f}, {np.max(phi):.6f}]")

    assert result.converged, "Low-level API solver did not converge!"
    assert angular.shape == (4, 50), f"Unexpected angular flux shape: {angular.shape}"

    print("  Low-level API test PASSED")


if __name__ == '__main__':
    test_isotropic_source()
    test_pure_absorber()
    test_two_region()
    test_spatial_convergence()
    test_quadrature_convergence()
    test_low_level_api()
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
