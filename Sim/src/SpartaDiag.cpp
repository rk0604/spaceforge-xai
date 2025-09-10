#include "SpartaDiag.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

static inline std::string trim(std::string s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c){return !issp(c);} ));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c){return !issp(c);}).base(), s.end());
    return s;
}

static bool split_csv(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(trim(tok));
    return !out.empty();
}

std::optional<SpartaDiag> read_sparta_diag_csv(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) return std::nullopt;

    std::string line;
    if (!std::getline(in, line)) return std::nullopt; // header
    std::string last;
    while (std::getline(in, line)) if (!trim(line).empty()) last = line;
    if (last.empty()) return std::nullopt;

    std::vector<std::string> toks;
    if (!split_csv(last, toks)) return std::nullopt;

    // Expect: step,time,temp_K,density_m3
    if (toks.size() < 4) return std::nullopt;
    SpartaDiag d{};
    try {
        d.step       = std::stod(toks[0]);
        d.time_s     = std::stod(toks[1]);
        d.temp_K     = std::stod(toks[2]);
        d.density_m3 = std::stod(toks[3]);
    } catch (...) { return std::nullopt; }
    return d;
}
