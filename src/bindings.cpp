#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "quadrature.h"
#include "mesh.h"
#include "xsdata.h"
#include "solver.h"
#include "kinetics.h"
#include "quasistatic.h"

namespace py = pybind11;

PYBIND11_MODULE(_snsolver, m)
{
    m.doc() = "1D Slab Geometry SN Neutron Transport Solver";

    py::class_<GaussLegendreQuadrature>(m, "Quadrature")
        .def(py::init<int>(), py::arg("order"))
        .def_property_readonly("order", &GaussLegendreQuadrature::order)
        .def_property_readonly("mu", &GaussLegendreQuadrature::mu)
        .def_property_readonly("wt", &GaussLegendreQuadrature::wt);

    py::class_<Mesh1D>(m, "Mesh1D")
        .def(py::init<double, double, int>(),
             py::arg("x_min"), py::arg("x_max"), py::arg("n_cells"))
        .def_property_readonly("n_cells", &Mesh1D::n_cells)
        .def_property_readonly("x_min", &Mesh1D::x_min)
        .def_property_readonly("x_max", &Mesh1D::x_max)
        .def_property_readonly("edges", &Mesh1D::edges)
        .def_property_readonly("centers", &Mesh1D::centers)
        .def_property_readonly("widths", &Mesh1D::widths);

    py::class_<MaterialXS>(m, "MaterialXS")
        .def_readonly("sigma_t", &MaterialXS::sigma_t)
        .def_readonly("sigma_s", &MaterialXS::sigma_s)
        .def_readonly("nu_sigma_f", &MaterialXS::nu_sigma_f)
        .def_readonly("source", &MaterialXS::source);

    py::class_<XSData>(m, "XSData")
        .def(py::init<>())
        .def("set_material", (void (XSData::*)(int, double, double, double)) &XSData::set_material,
             py::arg("mat_id"), py::arg("sigma_t"), py::arg("sigma_s"), py::arg("source"))
        .def("set_material_nuf", (void (XSData::*)(int, double, double, double, double)) &XSData::set_material,
             py::arg("mat_id"), py::arg("sigma_t"), py::arg("sigma_s"), py::arg("nu_sigma_f"), py::arg("source"))
        .def("set_cell_material", &XSData::set_cell_material,
             py::arg("cell"), py::arg("mat_id"))
        .def("set_cell_xs", (void (XSData::*)(int, double, double, double)) &XSData::set_cell_xs,
             py::arg("cell"), py::arg("sigma_t"), py::arg("sigma_s"), py::arg("source"))
        .def("set_cell_xs_nuf", (void (XSData::*)(int, double, double, double, double)) &XSData::set_cell_xs,
             py::arg("cell"), py::arg("sigma_t"), py::arg("sigma_s"), py::arg("nu_sigma_f"), py::arg("source"))
        .def("set_cell_nu_sigma_f", &XSData::set_cell_nu_sigma_f,
             py::arg("cell"), py::arg("nu_sigma_f"))
        .def("load_from_file", &XSData::load_from_file, py::arg("filename"))
        .def("finalize", &XSData::finalize, py::arg("n_cells"))
        .def_property_readonly("n_materials", &XSData::n_materials)
        .def_property_readonly("n_cells", &XSData::n_cells)
        .def("sigma_t", &XSData::sigma_t, py::arg("cell"))
        .def("sigma_s", &XSData::sigma_s, py::arg("cell"))
        .def("nu_sigma_f", &XSData::nu_sigma_f, py::arg("cell"))
        .def("source", &XSData::source, py::arg("cell"))
        .def_property_readonly("sigma_t_vec", &XSData::sigma_t_vec)
        .def_property_readonly("sigma_s_vec", &XSData::sigma_s_vec)
        .def_property_readonly("nu_sigma_f_vec", &XSData::nu_sigma_f_vec)
        .def_property_readonly("source_vec", &XSData::source_vec);

    py::class_<SolverResult>(m, "SolverResult")
        .def_readonly("scalar_flux", &SolverResult::scalar_flux)
        .def_readonly("iterations", &SolverResult::iterations)
        .def_readonly("error", &SolverResult::error)
        .def_readonly("converged", &SolverResult::converged);

    py::class_<SNSolver>(m, "SNSolver")
        .def(py::init<const GaussLegendreQuadrature&, const Mesh1D&, const XSData&, double, int>(),
             py::arg("quad"), py::arg("mesh"), py::arg("xs"),
             py::arg("tol") = 1e-6, py::arg("max_iter") = 10000)
        .def("solve", &SNSolver::solve)
        .def("solve_eigenvalue", &SNSolver::solve_eigenvalue,
             py::arg("initial_keff") = 1.0)
        .def("set_left_bc", &SNSolver::set_left_bc, py::arg("psi"))
        .def("set_right_bc", &SNSolver::set_right_bc, py::arg("psi"))
        .def("set_tolerance", &SNSolver::set_tolerance, py::arg("tol"))
        .def("set_max_iterations", &SNSolver::set_max_iterations, py::arg("max_iter"))
        .def("set_dsa_enabled", &SNSolver::set_dsa_enabled, py::arg("enabled"))
        .def("set_dsa_omega", &SNSolver::set_dsa_omega, py::arg("omega"))
        .def_property_readonly("dsa_enabled", &SNSolver::dsa_enabled)
        .def_property_readonly("dsa_omega", &SNSolver::dsa_omega)
        .def_property_readonly("scalar_flux", &SNSolver::scalar_flux)
        .def_property_readonly("angular_flux", &SNSolver::angular_flux)
        .def_property_readonly("keff", &SNSolver::keff);

    py::class_<DelayedNeutronData>(m, "DelayedNeutronData")
        .def(py::init<>())
        .def_property_readonly("n_groups", [](const DelayedNeutronData& d) { return d.n_groups; })
        .def_property_readonly("beta", [](const DelayedNeutronData& d) {
            return std::vector<double>(d.beta, d.beta + d.n_groups);
        })
        .def_property_readonly("lambda", [](const DelayedNeutronData& d) {
            return std::vector<double>(d.lambda, d.lambda + d.n_groups);
        })
        .def("beta_total", &DelayedNeutronData::beta_total)
        .def("set_beta", &DelayedNeutronData::set_beta, py::arg("i"), py::arg("val"))
        .def("set_lambda", &DelayedNeutronData::set_lambda, py::arg("i"), py::arg("val"))
        .def("get_beta", &DelayedNeutronData::get_beta, py::arg("i"))
        .def("get_lambda", &DelayedNeutronData::get_lambda, py::arg("i"));

    py::class_<KineticsResult>(m, "KineticsResult")
        .def_readonly("time", &KineticsResult::time)
        .def_readonly("power", &KineticsResult::power)
        .def_readonly("precursor_conc", &KineticsResult::precursor_conc)
        .def_readonly("reactivity", &KineticsResult::reactivity)
        .def_readonly("final_time", &KineticsResult::final_time)
        .def_readonly("n_steps", &KineticsResult::n_steps);

    py::class_<PointKinetics>(m, "PointKinetics")
        .def(py::init<const DelayedNeutronData&, double, double, double>(),
             py::arg("dn_data"), py::arg("Lambda"),
             py::arg("initial_power") = 1.0, py::arg("initial_source") = 0.0)
        .def("set_reactivity", &PointKinetics::set_reactivity, py::arg("rho"))
        .def("set_source", &PointKinetics::set_source, py::arg("S"))
        .def("set_reactivity_function", &PointKinetics::set_reactivity_function, py::arg("rho_func"))
        .def("solve", &PointKinetics::solve,
             py::arg("t_end"), py::arg("dt"), py::arg("theta") = 1.0)
        .def_property_readonly("power", &PointKinetics::power)
        .def_property_readonly("reactivity", &PointKinetics::reactivity)
        .def_property_readonly("precursors", &PointKinetics::precursors)
        .def_property_readonly("dn_data", &PointKinetics::dn_data)
        .def_property_readonly("generation_time", &PointKinetics::generation_time);

    py::class_<XSChange>(m, "XSChange")
        .def(py::init<>())
        .def(py::init<int, double, double, double, double>(),
             py::arg("cell"), py::arg("sigma_t"), py::arg("sigma_s"),
             py::arg("nu_sigma_f"), py::arg("source"))
        .def_readwrite("cell", &XSChange::cell)
        .def_readwrite("sigma_t", &XSChange::sigma_t)
        .def_readwrite("sigma_s", &XSChange::sigma_s)
        .def_readwrite("nu_sigma_f", &XSChange::nu_sigma_f)
        .def_readwrite("source", &XSChange::source);

    py::class_<QuasiStaticResult>(m, "QuasiStaticResult")
        .def_readonly("time", &QuasiStaticResult::time)
        .def_readonly("amplitude", &QuasiStaticResult::amplitude)
        .def_readonly("reactivity", &QuasiStaticResult::reactivity)
        .def_readonly("beta_eff", &QuasiStaticResult::beta_eff)
        .def_readonly("gen_time", &QuasiStaticResult::gen_time)
        .def_readonly("shape_flux", &QuasiStaticResult::shape_flux)
        .def_readonly("full_flux", &QuasiStaticResult::full_flux)
        .def_readonly("n_shape_updates", &QuasiStaticResult::n_shape_updates)
        .def_readonly("total_kinetics_steps", &QuasiStaticResult::total_kinetics_steps)
        .def_readonly("initial_keff", &QuasiStaticResult::initial_keff);

    py::class_<QuasiStaticSolver>(m, "QuasiStaticSolver")
        .def(py::init<const GaussLegendreQuadrature&, const Mesh1D&, const XSData&,
                      const DelayedNeutronData&, double, double, double, int>(),
             py::arg("quad"), py::arg("mesh"), py::arg("xs"),
             py::arg("dn_data"), py::arg("Lambda"), py::arg("nu_sigma_f_ref"),
             py::arg("tol") = 1e-6, py::arg("max_iter") = 1000)
        .def("set_left_bc", &QuasiStaticSolver::set_left_bc, py::arg("psi"))
        .def("set_right_bc", &QuasiStaticSolver::set_right_bc, py::arg("psi"))
        .def("set_xs_change_function", &QuasiStaticSolver::set_xs_change_function, py::arg("func"))
        .def("set_dsa_enabled", &QuasiStaticSolver::set_dsa_enabled, py::arg("enabled"))
        .def("set_eigenvalue_mode", &QuasiStaticSolver::set_eigenvalue_mode, py::arg("use_eigenvalue"))
        .def("solve", &QuasiStaticSolver::solve,
             py::arg("t_end"), py::arg("dt_shape"), py::arg("dt_kinetics"),
             py::arg("initial_power") = 1.0, py::arg("theta_kinetics") = 1.0)
        .def_property_readonly("shape", &QuasiStaticSolver::shape);
}
