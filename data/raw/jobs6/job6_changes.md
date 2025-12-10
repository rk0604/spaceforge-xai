# Run Metadata — jobs6

**Run ID:** jobs6  
**Directory:** data/raw/jobs6/  
**Local Time:** 2025-12-08 19:00  
**UTC Time:** 2025-12-09 00:00  

---

## 1. Simulation Configuration (High-Level)

| Setting | Value |
|--------|-------|
| Simulation mode (`--mode`) | wake |
| ENABLE_SPARTA | OFF |
| GPU mode (`GPU`) | OFF |
| MPI ranks (`NP`) | 1 |
| Wake deck (`WAKE_DECK`) | in.wake_harness |
| Input subdir | Sim/ |

**Purpose of this run:**  
Dataset creation for ST-GNN surrogate modeling after updating gate parameters (more realistic flux/temp tolerances). jobs6 reflects the new gating regime.

---

## 2. Orbit & Environment Parameters

| Variable | Meaning | Value |
|----------|----------|--------|
| PTORR_TARGET | Freestream ambient pressure [Torr] | 1e-7 |
| PCUP_TORR | Wake pressure at cupola [Torr] | 9e-9 |
| FWAfer_CM2S | Wafer-center flux [cm^-2 s^-1] | varies per job |
| CUP_BASE_SCALE | Base outgassing scale | 1.0 |
| CUP_AMP_SCALE | Outgassing amplitude | 0.5 |
| CUP_PHASE0 | Phase offset | 0.0 |

Additional physical parameters:  
- Altitude: 400 km  
- Temperature: ~800 K  
- Velocity: 7500 m/s  

---

## 3. Harness Timing / Coupling Configuration

| Parameter | Description | Value |
|----------|-------------|--------|
| nticks | Total harness ticks | 700+ |
| dt | Seconds per tick | 0.1 |
| sparta_block | Ticks between SPARTA calls | 2500 (unused since SPARTA OFF) |
| couple_every | Coupling frequency | 1 |

**Extra CLI flags:**  
```
--nticks 2600 --couple-every 1 --sparta-block 2500
```

---

## 4. Code + Deck Versions

| Item | Value |
|------|--------|
| Git repo | spaceforge-xai |
| Branch | main |
| Commit hash | UNKNOWN |
| Uncommitted changes? | yes — updated gate logic |
| SPARTA build used | none (ENABLE_SPARTA=OFF) |
| SPARTA executable | NONE |
| Important SPARTA flags | N/A |

---

## 5. Files in This Dataset

### simulationEngine.csv
- **Source:** C++ harness  
- **Description:** Per-tick simulation state  
- **One row represents:** 1 tick (0.1 s)  
- **Key columns:**  
  - tick — harness tick index  
  - wafer_flux_cm2s — commanded wafer flux  
  - P_actual_W — delivered heater power  
  - temp_proxy_K — RC model wafer temperature  
  - underflux_streak — consecutive ticks under flux tolerance  
  - temp_miss_streak — consecutive ticks under temp tolerance  
  - mbe_active — whether growth was active this tick  

### EffusionCell.csv
- **Source:** C++ harness  
- **Description:** Effusion cell thermal/flux response  
- **One row:** 1 tick  
- **Key columns:** P_actual_W, flux_atom_s, eff_temp_K, etc.

### wake_*.csv (if any)
- **Source:** SPARTA (not present in this run)

---

## 6. Observations, Anomalies, Notes

- First run using **realistic gates**:  
  - min_flux_fraction ≈ 0.93  
  - temp_tolerance_fraction ≈ 0.90  
  - underflux_limit_ticks ≈ 10  
  - temp_fail_limit_ticks ≈ 10  
- No SPARTA involvement for this run.  
- Dataset intended as clean training baseline for ST‑GNN.  
- No anomalies observed; job transitions and warm-up logic behaved normally.

