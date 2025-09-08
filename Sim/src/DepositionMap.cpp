#include "DepositionMap.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
constexpr double NM_PER_M = 1e9;

// Small binary header for save/load
struct BinHeader {
  char     magic[8];   // "DEPMAP\0"
  uint32_t version;    // 1
  int32_t  N;
  double   radius;
  uint64_t count;      // N*N (for sanity)
};

} // namespace

void DepositionMap::toThickness(double mass_per_particle_kg,
                                double rho_kg_per_m3,
                                double cellArea_m2) {
  // Converts bin "counts" to thickness [nm] per pixel using:
  // thickness_m = (#particles * mass_per_particle) / (rho * cellArea)
  // thickness_nm = thickness_m * 1e9
  if (mass_per_particle_kg <= 0 || rho_kg_per_m3 <= 0 || cellArea_m2 <= 0) {
    return;
  }
  const double factor_nm_per_count =
      (mass_per_particle_kg / (rho_kg_per_m3 * cellArea_m2)) * NM_PER_M;

  for (double &b : bins) {
    if (b <= 0.0) { b = 0.0; continue; }
    b *= factor_nm_per_count;
  }
}

void DepositionMap::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("DepositionMap::save: cannot open " + path);

  BinHeader h{};
  std::memset(&h, 0, sizeof(h));
  std::memcpy(h.magic, "DEPMAP\0", 8);
  h.version = 1;
  h.N       = static_cast<int32_t>(N);
  h.radius  = radius;
  h.count   = static_cast<uint64_t>(bins.size());

  out.write(reinterpret_cast<const char*>(&h), sizeof(h));
  out.write(reinterpret_cast<const char*>(bins.data()),
            static_cast<std::streamsize>(bins.size() * sizeof(double)));
  if (!out) throw std::runtime_error("DepositionMap::save: write failed");
}

bool DepositionMap::load(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  BinHeader h{};
  in.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!in) return false;
  if (std::memcmp(h.magic, "DEPMAP\0", 8) != 0 || h.version != 1) return false;
  if (h.N <= 0 || h.count != static_cast<uint64_t>(h.N) * static_cast<uint64_t>(h.N)) return false;

  N      = h.N;
  radius = h.radius;
  bins.assign(static_cast<size_t>(h.count), 0.0);

  in.read(reinterpret_cast<char*>(bins.data()),
          static_cast<std::streamsize>(bins.size() * sizeof(double)));
  if (!in) return false;
  return true;
}
