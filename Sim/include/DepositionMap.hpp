// DepositionMap.hpp
#pragma once
#include <vector>
#include <string>
#include <algorithm> // clamp, fill

struct DepositionMap {
  int N;                     // e.g. 64 (N x N grid)
  double radius;             // wafer radius (m), in wafer-local coords
  std::vector<double> bins;  // length N*N, counts or mass/area

  DepositionMap(int N_, double r)
  : N(N_), radius(r), bins(static_cast<size_t>(N_)*static_cast<size_t>(N_), 0.0) {}

  // Map (x,y) in wafer plane to a grid cell and accumulate 'w' (weight)
  inline void addHit(double x, double y, double w=1.0) {
    // normalize to [-1,1] using radius
    double nx = x / radius, ny = y / radius;
    if (nx*nx + ny*ny > 1.0) return;  // outside the disk
    int ix = static_cast<int>((nx*0.5 + 0.5) * N);
    int iy = static_cast<int>((ny*0.5 + 0.5) * N);

#if __cpp_lib_clamp >= 201603
    ix = std::clamp(ix, 0, N-1);
    iy = std::clamp(iy, 0, N-1);
#else
    if (ix < 0) ix = 0; else if (ix >= N) ix = N-1;
    if (iy < 0) iy = 0; else if (iy >= N) iy = N-1;
#endif

    bins[static_cast<size_t>(iy)*static_cast<size_t>(N) + static_cast<size_t>(ix)] += w;
  }

  void clear() { std::fill(bins.begin(), bins.end(), 0.0); }

  // Optional: convert counts to thickness (nm) using: thickness = (#atoms * m)/(rho*cell_area)
  void toThickness(double mass_per_particle, double rho, double cellArea);

  // Serialize/restore so you can pause/resume jobs
  void save(const std::string& path) const;
  bool load(const std::string& path);
};
