#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <limits>   // for quiet_NaN

struct SpartaDiag {
    double step{0};
    double time_s{0};
    double temp_K{std::numeric_limits<double>::quiet_NaN()};
    double density_m3{std::numeric_limits<double>::quiet_NaN()};
};

std::optional<SpartaDiag> read_sparta_diag_csv(const std::filesystem::path& file);

// Boltzmann constant [J/K]
constexpr double K_BOLTZ = 1.380649e-23;
