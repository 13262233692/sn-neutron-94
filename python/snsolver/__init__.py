from ._snsolver import (Quadrature, Mesh1D, XSData, MaterialXS, SNSolver, SolverResult,
                        DelayedNeutronData, PointKinetics, KineticsResult,
                        XSChange, QuasiStaticSolver, QuasiStaticResult)
import numpy as np


def solve_slab(sn_order, x_min, x_max, n_cells,
               sigma_t, sigma_s, source=None,
               left_bc=0.0, right_bc=0.0,
               tol=1e-6, max_iter=10000,
               dsa=True, dsa_omega=1.0):
    quad = Quadrature(sn_order)
    mesh = Mesh1D(x_min, x_max, n_cells)

    xs = XSData()
    if isinstance(sigma_t, (int, float)):
        sigma_t_arr = np.full(n_cells, float(sigma_t))
        sigma_s_arr = np.full(n_cells, float(sigma_s))
        source_arr = np.full(n_cells, float(source) if source is not None else 0.0)
    else:
        sigma_t_arr = np.asarray(sigma_t, dtype=float)
        sigma_s_arr = np.asarray(sigma_s, dtype=float)
        source_arr = np.asarray(source, dtype=float) if source is not None else np.zeros(n_cells)

    for i in range(n_cells):
        xs.set_cell_xs(i, sigma_t_arr[i], sigma_s_arr[i], source_arr[i])

    solver = SNSolver(quad, mesh, xs, tol, max_iter)
    solver.set_left_bc(left_bc)
    solver.set_right_bc(right_bc)
    solver.set_dsa_enabled(dsa)
    solver.set_dsa_omega(dsa_omega)

    result = solver.solve()

    return {
        'scalar_flux': np.array(result.scalar_flux),
        'iterations': result.iterations,
        'error': result.error,
        'converged': result.converged,
        'centers': np.array(mesh.centers),
        'mu': np.array(quad.mu),
        'wt': np.array(quad.wt),
        'angular_flux': np.array(solver.angular_flux),
    }


def solve_eigenvalue(sn_order, x_min, x_max, n_cells,
                     sigma_t, sigma_s, sigma_f, nu,
                     left_bc=0.0, right_bc=0.0,
                     tol=1e-6, max_iter=10000,
                     initial_keff=1.0,
                     dsa=True):
    quad = Quadrature(sn_order)
    mesh = Mesh1D(x_min, x_max, n_cells)

    xs = XSData()
    if isinstance(sigma_t, (int, float)):
        sigma_t_arr = np.full(n_cells, float(sigma_t))
        sigma_s_arr = np.full(n_cells, float(sigma_s))
        nu_sigma_f_arr = np.full(n_cells, float(nu * sigma_f))
    else:
        sigma_t_arr = np.asarray(sigma_t, dtype=float)
        sigma_s_arr = np.asarray(sigma_s, dtype=float)
        nu_sigma_f_arr = np.asarray(nu * np.asarray(sigma_f, dtype=float), dtype=float)

    for i in range(n_cells):
        xs.set_cell_xs(i,
                       sigma_t_arr[i] - sigma_s_arr[i] + nu_sigma_f_arr[i] / nu,
                       sigma_s_arr[i],
                       0.0)

    solver = SNSolver(quad, mesh, xs, tol, max_iter)
    solver.set_left_bc(left_bc)
    solver.set_right_bc(right_bc)
    solver.set_dsa_enabled(dsa)

    result = solver.solve_eigenvalue(initial_keff)

    return {
        'scalar_flux': np.array(result.scalar_flux),
        'keff': solver.keff,
        'iterations': result.iterations,
        'error': result.error,
        'converged': result.converged,
        'centers': np.array(mesh.centers),
        'mu': np.array(quad.mu),
        'wt': np.array(quad.wt),
    }


def load_and_solve(sn_order, x_min, x_max, n_cells,
                   xs_file, left_bc=0.0, right_bc=0.0,
                   tol=1e-6, max_iter=10000,
                   dsa=True):
    quad = Quadrature(sn_order)
    mesh = Mesh1D(x_min, x_max, n_cells)

    xs = XSData()
    xs.load_from_file(xs_file)
    if xs.n_cells() != n_cells:
        raise ValueError(f"XS file has {xs.n_cells()} cells, expected {n_cells}")

    solver = SNSolver(quad, mesh, xs, tol, max_iter)
    solver.set_left_bc(left_bc)
    solver.set_right_bc(right_bc)
    solver.set_dsa_enabled(dsa)

    result = solver.solve()

    return {
        'scalar_flux': np.array(result.scalar_flux),
        'iterations': result.iterations,
        'error': result.error,
        'converged': result.converged,
        'centers': np.array(mesh.centers),
    }


def solve_point_kinetics(rho, Lambda=None, beta_groups=None, lambda_groups=None,
                         initial_power=1.0, initial_source=0.0,
                         t_end=1.0, dt=0.001, theta=1.0):
    dn = DelayedNeutronData()
    if beta_groups is not None:
        for i, b in enumerate(beta_groups):
            dn.set_beta(i, b)
    if lambda_groups is not None:
        for i, l in enumerate(lambda_groups):
            dn.set_lambda(i, l)
    if Lambda is None:
        Lambda = 1e-4

    pk = PointKinetics(dn, Lambda, initial_power, initial_source)

    if callable(rho):
        pk.set_reactivity_function(rho)
    else:
        pk.set_reactivity(rho)

    result = pk.solve(t_end, dt, theta)

    return {
        'time': np.array(result.time),
        'power': np.array(result.power),
        'reactivity': np.array(result.reactivity),
        'precursor_conc': np.array(result.precursor_conc),
        'n_steps': result.n_steps,
        'beta_total': dn.beta_total(),
        'beta': np.array([dn.get_beta(i) for i in range(dn.n_groups)]),
        'lambda': np.array([dn.get_lambda(i) for i in range(dn.n_groups)]),
    }


def solve_quasistatic(sn_order, x_min, x_max, n_cells,
                      sigma_t, sigma_s, source=None,
                      nu_sigma_f=None, Lambda=1e-4,
                      rho_func=None, xs_change_func=None,
                      left_bc=0.0, right_bc=0.0,
                      t_end=1.0, dt_shape=0.1, dt_kinetics=0.001,
                      initial_power=1.0, theta_kinetics=1.0,
                      tol=1e-6, max_iter=1000, dsa=True,
                      eigenvalue_mode=False):
    quad = Quadrature(sn_order)
    mesh = Mesh1D(x_min, x_max, n_cells)

    xs = XSData()
    if isinstance(sigma_t, (int, float)):
        sigma_t_arr = np.full(n_cells, float(sigma_t))
        sigma_s_arr = np.full(n_cells, float(sigma_s))
        source_arr = np.full(n_cells, float(source) if source is not None else 0.0)
        if nu_sigma_f is None:
            nu_sigma_f_arr = sigma_t_arr - sigma_s_arr
        elif isinstance(nu_sigma_f, (int, float)):
            nu_sigma_f_arr = np.full(n_cells, float(nu_sigma_f))
        else:
            nu_sigma_f_arr = np.asarray(nu_sigma_f, dtype=float)
    else:
        sigma_t_arr = np.asarray(sigma_t, dtype=float)
        sigma_s_arr = np.asarray(sigma_s, dtype=float)
        source_arr = np.asarray(source, dtype=float) if source is not None else np.zeros(n_cells)
        if nu_sigma_f is None:
            nu_sigma_f_arr = sigma_t_arr - sigma_s_arr
        else:
            nu_sigma_f_arr = np.asarray(nu_sigma_f, dtype=float)

    for i in range(n_cells):
        xs.set_cell_xs_nuf(i, sigma_t_arr[i], sigma_s_arr[i], nu_sigma_f_arr[i], source_arr[i])

    dn = DelayedNeutronData()
    nu_sigma_f_ref = float(np.mean(nu_sigma_f_arr))

    qs = QuasiStaticSolver(quad, mesh, xs, dn, Lambda, nu_sigma_f_ref, tol, max_iter)
    qs.set_left_bc(left_bc)
    qs.set_right_bc(right_bc)
    qs.set_dsa_enabled(dsa)
    qs.set_eigenvalue_mode(eigenvalue_mode)

    if xs_change_func is not None:
        qs.set_xs_change_function(xs_change_func)

    result = qs.solve(t_end, dt_shape, dt_kinetics, initial_power, theta_kinetics)

    return {
        'time': np.array(result.time),
        'amplitude': np.array(result.amplitude),
        'reactivity': np.array(result.reactivity),
        'beta_eff': np.array(result.beta_eff),
        'gen_time': np.array(result.gen_time),
        'full_flux': [np.array(f) for f in result.full_flux],
        'shape_flux': [np.array(f) for f in result.shape_flux],
        'centers': np.array(mesh.centers),
        'n_shape_updates': result.n_shape_updates,
        'total_kinetics_steps': result.total_kinetics_steps,
        'initial_keff': result.initial_keff,
    }
