#pragma once

#include <vector>
#include <string>

struct MaterialXS {
    double sigma_t;
    double sigma_s;
    double nu_sigma_f;
    double source;

    MaterialXS() : sigma_t(0), sigma_s(0), nu_sigma_f(0), source(0) {}
};

class XSData {
public:
    XSData() = default;

    void set_material(int mat_id, double sigma_t, double sigma_s, double source);
    void set_material(int mat_id, double sigma_t, double sigma_s, double nu_sigma_f, double source);
    void set_cell_material(int cell, int mat_id);
    void set_cell_xs(int cell, double sigma_t, double sigma_s, double source);
    void set_cell_xs(int cell, double sigma_t, double sigma_s, double nu_sigma_f, double source);
    void set_cell_nu_sigma_f(int cell, double nu_sigma_f);
    void load_from_file(const std::string& filename);

    int n_materials() const { return static_cast<int>(materials_.size()); }
    const MaterialXS& material(int id) const { return materials_.at(id); }

    int n_cells() const { return static_cast<int>(cell_sigma_t_.size()); }

    double sigma_t(int cell) const { return cell_sigma_t_[cell]; }
    double sigma_s(int cell) const { return cell_sigma_s_[cell]; }
    double nu_sigma_f(int cell) const;
    double source(int cell) const { return cell_source_[cell]; }

    const std::vector<double>& sigma_t_vec() const { return cell_sigma_t_; }
    const std::vector<double>& sigma_s_vec() const { return cell_sigma_s_; }
    const std::vector<double>& nu_sigma_f_vec() const { return cell_nu_sigma_f_; }
    const std::vector<double>& source_vec() const { return cell_source_; }

    void finalize(int n_cells);

private:
    std::vector<MaterialXS> materials_;
    std::vector<int> cell_material_;
    std::vector<double> cell_sigma_t_;
    std::vector<double> cell_sigma_s_;
    std::vector<double> cell_nu_sigma_f_;
    std::vector<double> cell_source_;
};
