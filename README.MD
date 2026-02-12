# Polymer Chain Simulation Analysis Report

**Date:** February 12, 2026

## 1. Extracted Scaling Exponents

| Observable                  | Exponent   | ± stderr   | Expected | Status            |
|-----------------------------|------------|------------|----------|-------------------|
| 2ν (from Rg²)               | 1.1249     | 0.0519     | 1.176    | OK (4.3% off)     |
| 2ν (from R²)                | 1.0253     | 0.0891     | 1.176    | OK (12.8% off)    |
| α (D ~ N^α, N≥50)           | -1.0821    | 0.1410     | -1.0     | OK (8.2% off)     |
| β (τ_R ~ N^β)               | 1.7330     | 2-pt only  | 2.18     | WARN (20.5% off)  |

## 2. R² / Rg² Ratios  
*(Theoretical value in good solvent ≈ 6.0)*

| N   | R²/Rg² | Status                  |
|-----|--------|-------------------------|
| 25  | 9.044  | FAIL (finite-size)      |
| 50  | 6.655  | PASS                    |
| 75  | 7.647  | FAIL                    |
| 100 | 6.009  | PASS                    |
| 200 | 6.498  | PASS                    |
| 400 | 6.568  | PASS                    |

## 3. Generated Figures

- **fig1_static_scaling.png**  
  R² and Rg² vs N (log-log) with power-law fits

- **fig2_g1_msd.png**  
  Monomer mean-squared displacement g₁(t) for all chain lengths

- **fig3_gcm_msd.png**  
  Center-of-mass mean-squared displacement g_CM(t) for all chain lengths

- **fig3_diffusion_scaling.png**  
  Diffusion coefficient D vs N (log-log), with N=25 marked as outlier

- **fig4_autocorr_fits.png**  
  6-panel end-to-end vector autocorrelation decay with exponential fits

- **fig4_relaxation_scaling.png**  
  Relaxation time τ_R vs N (using only N=200 and N=400 data points)

## 4. Generated Data Tables

- `fig1_scaling_summary.tsv`  
- `fig3_diffusion_summary.tsv`  
- `fig4_tau_summary.tsv`

## Key Findings & Discussion

- **Static scaling** is well reproduced, particularly when using radius of gyration Rg² (2ν = 1.125, only 4.3% below expected 1.176).  
  End-to-end distance R² shows more noise (likely due to larger fluctuations and only 20 independent chains), but is still within reasonable tolerance.

- **Diffusion scaling** follows D ∼ N⁻¹ within ~8% when excluding the shortest chain (N=25), which suffers from strong finite-size effects (R²/Rg² = 9.04 ≫ 6).

- **Relaxation time extraction** is the weakest part of the analysis:  
  - Chains with N ≤ 100 fully decorrelate within the 100k sweep simulation window → autocorrelation functions become noise-dominated.  
  - Only N=200 and N=400 retain a measurable decay envelope.  
  - The resulting two-point fit gives β ≈ 1.73 (20.5% below expected 2.18).  
  - N=400 chain (τ_R ≈ 67k sweeps) has likely not fully relaxed within the 100k sweep simulation time, contributing to the underestimate.

**Recommendation:**  
For reliable τ_R scaling, future runs should use significantly longer simulation times (≥ 5–10 × τ_R) for N ≥ 200, and/or include longer chains (N=800, 1600) if computational resources allow.
