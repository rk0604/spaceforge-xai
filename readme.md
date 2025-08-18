# spaceforge_sim — SPARTA (external) setup

This repo links to a **prebuilt external SPARTA** library. You build SPARTA once outside the repo, then compile this project against it.

---

## 0) Requirements (iLab/Linux)

- GCC/G++ and CMake
- OpenMPI (or your site’s MPI) with wrappers: `mpicc`, `mpicxx`
- Git

Check:
```bash
which mpicxx && mpicxx --version   # confirms MPI C++ wrapper exists and which compiler it uses
which cmake && cmake --version     # confirms CMake
----
### if MPIcxx is not found on your host