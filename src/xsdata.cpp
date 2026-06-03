#include "xsdata.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

void XSData::set_material(int mat_id, double sigma_t, double sigma_s, double source)
{
    if (mat_id >= static_cast<int>(materials_.size())) {
        materials_.resize(mat_id + 1);
    }
    materials_[mat_id].sigma_t = sigma_t;
    materials_[mat_id].sigma_s = sigma_s;
    materials_[mat_id].nu_sigma_f = sigma_t - sigma_s;
    materials_[mat_id].source = source;
}

void XSData::set_material(int mat_id, double sigma_t, double sigma_s, double nu_sigma_f, double source)
{
    if (mat_id >= static_cast<int>(materials_.size())) {
        materials_.resize(mat_id + 1);
    }
    materials_[mat_id].sigma_t = sigma_t;
    materials_[mat_id].sigma_s = sigma_s;
    materials_[mat_id].nu_sigma_f = nu_sigma_f;
    materials_[mat_id].source = source;
}

void XSData::set_cell_material(int cell, int mat_id)
{
    if (cell >= static_cast<int>(cell_material_.size())) {
        cell_material_.resize(cell + 1, -1);
    }
    cell_material_[cell] = mat_id;
}

void XSData::set_cell_xs(int cell, double sigma_t, double sigma_s, double source)
{
    if (cell >= static_cast<int>(cell_sigma_t_.size())) {
        cell_sigma_t_.resize(cell + 1, 0.0);
        cell_sigma_s_.resize(cell + 1, 0.0);
        cell_nu_sigma_f_.resize(cell + 1, 0.0);
        cell_source_.resize(cell + 1, 0.0);
    }
    cell_sigma_t_[cell] = sigma_t;
    cell_sigma_s_[cell] = sigma_s;
    cell_nu_sigma_f_[cell] = sigma_t - sigma_s;
    cell_source_[cell] = source;
}

void XSData::set_cell_xs(int cell, double sigma_t, double sigma_s, double nu_sigma_f, double source)
{
    if (cell >= static_cast<int>(cell_sigma_t_.size())) {
        cell_sigma_t_.resize(cell + 1, 0.0);
        cell_sigma_s_.resize(cell + 1, 0.0);
        cell_nu_sigma_f_.resize(cell + 1, 0.0);
        cell_source_.resize(cell + 1, 0.0);
    }
    cell_sigma_t_[cell] = sigma_t;
    cell_sigma_s_[cell] = sigma_s;
    cell_nu_sigma_f_[cell] = nu_sigma_f;
    cell_source_[cell] = source;
}

void XSData::set_cell_nu_sigma_f(int cell, double nu_sigma_f)
{
    if (cell < static_cast<int>(cell_nu_sigma_f_.size())) {
        cell_nu_sigma_f_[cell] = nu_sigma_f;
    }
}

double XSData::nu_sigma_f(int cell) const
{
    if (cell < static_cast<int>(cell_nu_sigma_f_.size()) && cell_nu_sigma_f_[cell] > 0.0) {
        return cell_nu_sigma_f_[cell];
    }
    return cell_sigma_t_[cell] - cell_sigma_s_[cell];
}

void XSData::finalize(int n_cells)
{
    cell_sigma_t_.resize(n_cells, 0.0);
    cell_sigma_s_.resize(n_cells, 0.0);
    cell_nu_sigma_f_.resize(n_cells, 0.0);
    cell_source_.resize(n_cells, 0.0);

    if (!cell_material_.empty()) {
        for (int i = 0; i < n_cells; ++i) {
            if (i < static_cast<int>(cell_material_.size()) && cell_material_[i] >= 0) {
                const auto& mat = materials_[cell_material_[i]];
                cell_sigma_t_[i] = mat.sigma_t;
                cell_sigma_s_[i] = mat.sigma_s;
                cell_nu_sigma_f_[i] = mat.nu_sigma_f;
                cell_source_[i] = mat.source;
            }
        }
    }
}

void XSData::load_from_file(const std::string& filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open XS file: " + filename);
    }

    int n_mats;
    ifs >> n_mats;
    materials_.resize(n_mats);

    for (int m = 0; m < n_mats; ++m) {
        ifs >> materials_[m].sigma_t >> materials_[m].sigma_s >> materials_[m].source;
        materials_[m].nu_sigma_f = materials_[m].sigma_t - materials_[m].sigma_s;
    }

    int n_cells;
    ifs >> n_cells;
    cell_material_.resize(n_cells);

    for (int c = 0; c < n_cells; ++c) {
        ifs >> cell_material_[c];
    }

    finalize(n_cells);
}
