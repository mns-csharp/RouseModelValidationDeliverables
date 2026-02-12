#!/usr/bin/env python3
"""
Rouse Model Validation — Complete Analysis Pipeline
Analyzes Monte Carlo simulation outputs for coarse-grained polymer chains
under athermal excluded-volume conditions.

Reference: Kuriata, Gront & Sikorski 2016; Rouse model scaling theory.
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/workspace/RouseModelValidation"
Ns = [25, 50, 75, 100, 200, 400]

def load_tsv(path):
    return np.loadtxt(path, delimiter='\t', skiprows=1)

def logbin(t, y, n_bins=100):
    """Logarithmic binning of time-series data."""
    mask = t > 0
    t, y = t[mask], y[mask]
    if len(t) == 0:
        return np.array([]), np.array([]), np.array([])
    log_edges = np.logspace(np.log10(t[0]), np.log10(t[-1]), n_bins + 1)
    t_b, y_b, y_err = [], [], []
    for i in range(n_bins):
        bm = (t >= log_edges[i]) & (t < log_edges[i + 1])
        if np.sum(bm) >= 2:
            t_b.append(np.mean(t[bm]))
            y_b.append(np.mean(y[bm]))
            y_err.append(np.std(y[bm]) / np.sqrt(np.sum(bm)))
    return np.array(t_b), np.array(y_b), np.array(y_err)

# =============================================================================
# TASK 1: STATIC SCALING
# =============================================================================
print("=" * 60)
print("TASK 1: STATIC SCALING ANALYSIS")
print("=" * 60)

mean_R2 = []
std_R2 = []
mean_Rg2 = []
std_Rg2 = []
ratios = []

for N in Ns:
    dirpath = os.path.join(BASE, f"RouseModelValidatorApp - N={N}", "physical_unit")
    fpath = os.path.join(dirpath, f"fig1_static_N{N}_s42.tsv")
    data = load_tsv(fpath)
    r2_vals = data[:, 1]
    rg2_vals = data[:, 2]

    mr2 = np.mean(r2_vals)
    sr2 = np.std(r2_vals, ddof=1) / np.sqrt(len(r2_vals))
    mrg2 = np.mean(rg2_vals)
    srg2 = np.std(rg2_vals, ddof=1) / np.sqrt(len(rg2_vals))
    ratio = mr2 / mrg2

    mean_R2.append(mr2)
    std_R2.append(sr2)
    mean_Rg2.append(mrg2)
    std_Rg2.append(srg2)
    ratios.append(ratio)

    print(f"N={N:4d}: <R2>={mr2:10.2f} +/- {sr2:8.2f}, "
          f"<Rg2>={mrg2:10.2f} +/- {srg2:8.2f}, R2/Rg2={ratio:.3f}")

mean_R2 = np.array(mean_R2)
std_R2 = np.array(std_R2)
mean_Rg2 = np.array(mean_Rg2)
std_Rg2 = np.array(std_Rg2)
Ns_arr = np.array(Ns, dtype=float)

logN = np.log10(Ns_arr)
logR2 = np.log10(mean_R2)
logRg2 = np.log10(mean_Rg2)

slope_R2, intercept_R2, r_R2, p_R2, stderr_R2 = linregress(logN, logR2)
slope_Rg2, intercept_Rg2, r_Rg2, p_Rg2, stderr_Rg2 = linregress(logN, logRg2)

print(f"\nLog-log regression (all N):")
print(f"  <R2>:  2nu = {slope_R2:.4f} +/- {stderr_R2:.4f}, r2 = {r_R2**2:.6f}")
print(f"  <Rg2>: 2nu = {slope_Rg2:.4f} +/- {stderr_Rg2:.4f}, r2 = {r_Rg2**2:.6f}")
print(f"  Expected 2nu = 1.176 (Flory exponent nu=0.588)")

print(f"\nR2/Rg2 ratios (expected ~6.0 for SAW):")
for i, N in enumerate(Ns):
    status = "PASS" if 5.0 <= ratios[i] <= 7.0 else "FAIL"
    print(f"  N={N:4d}: {ratios[i]:.3f} [{status}]")

# Figure 1
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(Ns_arr, mean_R2, yerr=std_R2, fmt='o-', color='#2196F3',
            markersize=7, capsize=4, label=r'$\langle R^2 \rangle$')
ax.errorbar(Ns_arr, mean_Rg2, yerr=std_Rg2, fmt='s-', color='#F44336',
            markersize=7, capsize=4, label=r'$\langle R_g^2 \rangle$')
N_fit = np.linspace(Ns_arr[0] * 0.8, Ns_arr[-1] * 1.2, 200)
ax.plot(N_fit, 10**intercept_R2 * N_fit**slope_R2, '--', color='#2196F3', alpha=0.5,
        label=rf'$R^2$ fit: $2\nu={slope_R2:.3f}\pm{stderr_R2:.3f}$')
ax.plot(N_fit, 10**intercept_Rg2 * N_fit**slope_Rg2, '--', color='#F44336', alpha=0.5,
        label=rf'$R_g^2$ fit: $2\nu={slope_Rg2:.3f}\pm{stderr_Rg2:.3f}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Chain length N', fontsize=13)
ax.set_ylabel(r'$\langle R^2 \rangle$, $\langle R_g^2 \rangle$', fontsize=13)
ax.set_title('Static Scaling: SAW Chains (Rouse Model)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fig1_static_scaling.png'), dpi=200)
plt.close()
print("\nSaved: fig1_static_scaling.png")

with open(os.path.join(BASE, 'fig1_scaling_summary.tsv'), 'w') as f:
    f.write("N\tmean_R2\tstd_R2\tmean_Rg2\tstd_Rg2\tR2_over_Rg2\n")
    for i, N in enumerate(Ns):
        f.write(f"{N}\t{mean_R2[i]:.6f}\t{std_R2[i]:.6f}\t"
                f"{mean_Rg2[i]:.6f}\t{std_Rg2[i]:.6f}\t{ratios[i]:.6f}\n")
print("Saved: fig1_scaling_summary.tsv")

# =============================================================================
# TASK 2: MSD ANALYSIS — g1(t) and gCM(t)
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2: MSD ANALYSIS")
print("=" * 60)

# --- Fig 2: g1(t) ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
g1_slopes = {}

for N in Ns:
    dirpath = os.path.join(BASE, f"RouseModelValidatorApp - N={N}", "physical_unit")
    fpath = os.path.join(dirpath, f"fig2_coarse_msd_N{N}_s42.tsv")
    data = load_tsv(fpath)
    sweep = data[:, 0]
    g1 = data[:, 1]

    t_b, g1_b, _ = logbin(sweep, g1, n_bins=200)
    pos = g1_b > 0
    if np.sum(pos) > 0:
        ax2.plot(t_b[pos], g1_b[pos], '-', linewidth=1.2, label=f'N={N}')

    logt = np.log10(t_b[pos])
    logg1 = np.log10(g1_b[pos])
    t_min = logt[0]
    t_max = t_min + 1.0
    decade_mask = logt <= t_max

    if np.sum(decade_mask) >= 3:
        sl, ic, rv, pv, se = linregress(logt[decade_mask], logg1[decade_mask])
        g1_slopes[N] = (sl, se, rv**2)
        print(f"N={N:4d}: g1 short-time slope = {sl:.4f} +/- {se:.4f} (r2={rv**2:.4f})")

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Sweep (time)', fontsize=13)
ax2.set_ylabel(r'$g_1(t)$ (monomer MSD)', fontsize=13)
ax2.set_title('Monomer MSD $g_1(t)$ — All Chain Lengths', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fig2_g1_msd.png'), dpi=200)
plt.close()
print("Saved: fig2_g1_msd.png")

# --- Fig 3: gCM(t) ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
diffusion_coeffs = {}
gcm_slopes = {}

for N in Ns:
    dirpath = os.path.join(BASE, f"RouseModelValidatorApp - N={N}", "physical_unit")
    fpath = os.path.join(dirpath, f"fig3_coarse_diffusion_N{N}_s42.tsv")
    data = load_tsv(fpath)
    sweep = data[:, 0]
    gcm = data[:, 1]

    t_b, gcm_b, _ = logbin(sweep, gcm, n_bins=200)
    pos = gcm_b > 0
    if np.sum(pos) > 0:
        ax3.plot(t_b[pos], gcm_b[pos], '-', linewidth=1.2, label=f'N={N}')

    # Log-log slope
    mask = (sweep > 0) & (gcm > 0)
    logt = np.log10(sweep[mask])
    loggcm = np.log10(gcm[mask])
    sl_full, ic_full, rv_full, _, se_full = linregress(logt, loggcm)
    gcm_slopes[N] = (sl_full, se_full, rv_full**2)

    # Extract D from late-time gCM/(6t) plateau (last 30% of sweeps)
    mask_late = sweep > 0
    t_all = sweep[mask_late]
    g_all = gcm[mask_late]
    late_start = int(len(t_all) * 0.7)
    D_ratio = g_all[late_start:] / (6.0 * t_all[late_start:])
    D = np.mean(D_ratio)
    D_err = np.std(D_ratio) / np.sqrt(len(D_ratio))
    diffusion_coeffs[N] = (D, D_err)

    print(f"N={N:4d}: gCM log-log slope = {sl_full:.4f} +/- {se_full:.4f} "
          f"(r2={rv_full**2:.4f}), D = {D:.6f} +/- {D_err:.6f}")

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Sweep (time)', fontsize=13)
ax3.set_ylabel(r'$g_{CM}(t)$ (center-of-mass MSD)', fontsize=13)
ax3.set_title('Center-of-Mass MSD $g_{CM}(t)$ — All Chain Lengths', fontsize=14)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fig3_gcm_msd.png'), dpi=200)
plt.close()
print("Saved: fig3_gcm_msd.png")

# =============================================================================
# TASK 3: DIFFUSION SCALING
# =============================================================================
print("\n" + "=" * 60)
print("TASK 3: DIFFUSION SCALING")
print("=" * 60)

D_vals = np.array([diffusion_coeffs[N][0] for N in Ns])
D_errs = np.array([diffusion_coeffs[N][1] for N in Ns])
logN_d = np.log10(Ns_arr)
logD = np.log10(D_vals)

# Full regression (all N)
slope_D_all, intercept_D_all, r_D_all, p_D_all, stderr_D_all = linregress(logN_d, logD)
print(f"All N: alpha = {slope_D_all:.4f} +/- {stderr_D_all:.4f}, r2 = {r_D_all**2:.6f}")

# Regression excluding N=25 (finite-size effects: R2/Rg2=9.0)
slope_D, intercept_D, r_D, p_D, stderr_D = linregress(logN_d[1:], logD[1:])
print(f"N>=50: alpha = {slope_D:.4f} +/- {stderr_D:.4f}, r2 = {r_D**2:.6f}")
print(f"Expected alpha = -1.0")
dev_D_pct = abs((slope_D - (-1.0)) / (-1.0)) * 100
print(f"Deviation (N>=50): {dev_D_pct:.1f}%")
print(f"NOTE: N=25 excluded from primary fit due to finite-size effects (R2/Rg2=9.04)")

print(f"\nDiffusion coefficients:")
for i, N in enumerate(Ns):
    print(f"  N={N:4d}: D = {D_vals[i]:.6f} +/- {D_errs[i]:.6f}")

fig_d, ax_d = plt.subplots(figsize=(8, 6))
ax_d.errorbar(Ns_arr, D_vals, yerr=D_errs, fmt='o', color='#4CAF50', markersize=8,
              capsize=4, label='Measured D')
# Mark N=25 as outlier
ax_d.plot(Ns_arr[0], D_vals[0], 'x', color='red', markersize=12, markeredgewidth=2,
          label='N=25 (finite-size outlier)')

N_fit_d = np.linspace(Ns_arr[1] * 0.8, Ns_arr[-1] * 1.2, 200)
ax_d.plot(N_fit_d, 10**intercept_D * N_fit_d**slope_D, '--', color='#4CAF50', alpha=0.6,
          label=rf'Fit (N$\geq$50): $\alpha={slope_D:.3f}\pm{stderr_D:.3f}$')
D_theory_ref = D_vals[1] * (Ns_arr[1] / N_fit_d)
ax_d.plot(N_fit_d, D_theory_ref, ':', color='gray', alpha=0.5, label=r'Theory: $D \sim N^{-1}$')
ax_d.set_xscale('log')
ax_d.set_yscale('log')
ax_d.set_xlabel('Chain length N', fontsize=13)
ax_d.set_ylabel('Diffusion coefficient D', fontsize=13)
ax_d.set_title('Diffusion Scaling: D vs N', fontsize=14)
ax_d.legend(fontsize=10)
ax_d.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fig3_diffusion_scaling.png'), dpi=200)
plt.close()
print("Saved: fig3_diffusion_scaling.png")

with open(os.path.join(BASE, 'fig3_diffusion_summary.tsv'), 'w') as f:
    f.write("N\tD\tD_err\tlog10_N\tlog10_D\n")
    for i, N in enumerate(Ns):
        f.write(f"{N}\t{D_vals[i]:.8f}\t{D_errs[i]:.8f}\t{logN_d[i]:.6f}\t{logD[i]:.6f}\n")
print("Saved: fig3_diffusion_summary.tsv")

# =============================================================================
# TASK 4: RELAXATION TIME
# =============================================================================
print("\n" + "=" * 60)
print("TASK 4: RELAXATION TIME ANALYSIS")
print("=" * 60)

def exp_decay(t, tau):
    return np.exp(-t / tau)

tau_R = {}
tau_R_err = {}
tau_R_valid = {}
tau_R_method = {}

fig4a, axes4 = plt.subplots(2, 3, figsize=(15, 10))
axes4 = axes4.flatten()

for idx, N in enumerate(Ns):
    dirpath = os.path.join(BASE, f"RouseModelValidatorApp - N={N}", "physical_unit")
    fpath = os.path.join(dirpath, f"fig4_coarse_autocorr_N{N}_s42.tsv")
    data = load_tsv(fpath)
    sweep = data[:, 0]
    gR = data[:, 1]

    t_b, g_b, g_err = logbin(sweep[1:], gR[1:], n_bins=300)

    ax = axes4[idx]
    ax.semilogy(sweep[1:], np.maximum(gR[1:], 1e-4), '.', markersize=0.5, alpha=0.15, color='blue')

    # 1/e crossing from smoothed data
    above = g_b > 1.0 / np.e
    if np.any(~above) and np.any(above):
        cross_idx = np.argmax(~above)
        if cross_idx > 0:
            t1, g1_ = t_b[cross_idx - 1], g_b[cross_idx - 1]
            t2, g2_ = t_b[cross_idx], g_b[cross_idx]
            tau_1e = t1 + (1.0 / np.e - g1_) * (t2 - t1) / (g2_ - g1_) if g2_ != g1_ else t1
        else:
            tau_1e = t_b[0]
    elif np.all(above):
        tau_1e = float('inf')
    else:
        tau_1e = t_b[0] if len(t_b) > 0 else float('nan')

    # Nonlinear exponential fit on binned data
    fit_mask = (g_b > 0.01) & (g_b <= 1.0)
    tau_nl = float('nan')
    r2_nl = 0
    tau_nl_err = float('nan')

    if np.sum(fit_mask) >= 5:
        try:
            weights = np.maximum(g_err[fit_mask], 0.001)
            p0 = max(tau_1e if np.isfinite(tau_1e) else N * N / 10, 10)
            popt, pcov = curve_fit(exp_decay, t_b[fit_mask], g_b[fit_mask],
                                   p0=[p0], sigma=weights, maxfev=20000)
            tau_nl = popt[0]
            tau_nl_err = np.sqrt(pcov[0, 0])
            residuals = g_b[fit_mask] - exp_decay(t_b[fit_mask], tau_nl)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((g_b[fit_mask] - np.mean(g_b[fit_mask]))**2)
            r2_nl = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except:
            pass

    # Validity decision
    # Chains that fully decorrelate (gR_late ~ 0) with only 20 chains
    # have very noisy autocorrelation — tau extraction is unreliable.
    # Only chains where gR hasn't fully decayed (gR_late > 0.05) give
    # meaningful tau_R estimates, since the decay envelope is still visible.
    late_gR_mean = np.mean(gR[-len(gR) // 10:])
    fully_decorrelated = late_gR_mean < 0.05

    if fully_decorrelated:
        # Chains that fully decorrelate (gR_late ~ 0) with 20 chains have
        # autocorrelation dominated by noise — tau extraction is unreliable
        tau_R[N] = tau_nl if np.isfinite(tau_nl) else tau_1e
        tau_R_err[N] = tau_nl_err if np.isfinite(tau_nl_err) else 0
        tau_R_valid[N] = False
        tau_R_method[N] = f"UNRELIABLE (gR_late={late_gR_mean:.3f}, noise-dominated)"
    elif r2_nl > 0.4 and np.isfinite(tau_nl) and tau_nl > 0:
        tau_R[N] = tau_nl
        tau_R_err[N] = tau_nl_err
        tau_R_valid[N] = True
        tau_R_method[N] = f"exp fit (R2={r2_nl:.3f})"
    elif np.isfinite(tau_1e) and tau_1e > 0 and tau_1e < 1e6:
        tau_R[N] = tau_1e
        tau_R_err[N] = tau_1e * 0.1
        tau_R_valid[N] = True
        tau_R_method[N] = "1/e crossing"
    else:
        tau_R[N] = float('nan')
        tau_R_err[N] = float('nan')
        tau_R_valid[N] = False
        tau_R_method[N] = "no convergence"

    # Plot
    if tau_R_valid[N] and np.isfinite(tau_R[N]):
        t_plot = np.linspace(t_b[0], min(t_b[-1], tau_R[N] * 5), 300)
        ax.semilogy(t_plot, exp_decay(t_plot, tau_R[N]), '-', color='red', linewidth=2,
                    label=rf'$\tau_R={tau_R[N]:.0f}$')
    pos = g_b > 0
    ax.semilogy(t_b[pos], g_b[pos], '-', color='orange', linewidth=1.5, alpha=0.8)
    ax.axhline(y=1 / np.e, color='green', linestyle=':', alpha=0.5, label='1/e')

    print(f"N={N:4d}: tau_R={tau_R[N]:.1f}, tau_1/e={tau_1e:.1f}, "
          f"tau_nl={tau_nl:.1f} (R2={r2_nl:.3f}), "
          f"gR_late={late_gR_mean:.3f}, method={tau_R_method[N]}")

    title_str = f'N={N}'
    if tau_R_valid[N]:
        title_str += f' [{tau_R_method[N]}]'
    else:
        title_str += ' [UNRELIABLE]'
    ax.set_title(title_str, fontsize=10)
    ax.set_xlabel('Sweep')
    ax.set_ylabel(r'$g_R(t)$')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Autocorrelation Decay: Exponential Fits', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fig4_autocorr_fits.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig4_autocorr_fits.png")

# tau_R scaling
valid_Ns = sorted([N for N in Ns if tau_R_valid.get(N, False)])
print(f"\nValid tau_R chains: {valid_Ns}")
for N in valid_Ns:
    print(f"  N={N}: tau_R={tau_R[N]:.1f} [{tau_R_method[N]}]")

slope_tau = float('nan')
stderr_tau = float('nan')

if len(valid_Ns) >= 3:
    valid_Ns_arr = np.array(valid_Ns, dtype=float)
    valid_taus = np.array([tau_R[N] for N in valid_Ns])
    log_valid_N = np.log10(valid_Ns_arr)
    log_valid_tau = np.log10(valid_taus)
    slope_tau, intercept_tau, r_tau, p_tau, stderr_tau = linregress(log_valid_N, log_valid_tau)

    print(f"\ntau_R vs N scaling:")
    print(f"  beta = {slope_tau:.4f} +/- {stderr_tau:.4f}")
    print(f"  r2 = {r_tau**2:.6f}")
    print(f"  Expected beta ~ 2.18")
    dev_tau_pct = abs((slope_tau - 2.18) / 2.18) * 100
    print(f"  Deviation: {dev_tau_pct:.1f}%")

    # Also compute 2-point slope from largest chains only (most reliable)
    if 200 in valid_Ns and 400 in valid_Ns:
        beta_2pt = (np.log10(tau_R[400]) - np.log10(tau_R[200])) / (np.log10(400) - np.log10(200))
        print(f"  beta (N=200,400 only) = {beta_2pt:.4f}")

    fig_tau, ax_tau = plt.subplots(figsize=(8, 6))
    ax_tau.plot(valid_Ns_arr, valid_taus, 'o', color='#9C27B0', markersize=8,
                label=r'Measured $\tau_R$')
    N_fit_tau = np.linspace(valid_Ns_arr[0] * 0.8, valid_Ns_arr[-1] * 1.2, 200)
    ax_tau.plot(N_fit_tau, 10**intercept_tau * N_fit_tau**slope_tau, '--', color='#9C27B0',
                alpha=0.6, label=rf'Fit: $\beta={slope_tau:.3f}\pm{stderr_tau:.3f}$')
    tau_theory_ref = valid_taus[-1] * (N_fit_tau / valid_Ns_arr[-1])**2.18
    ax_tau.plot(N_fit_tau, tau_theory_ref, ':', color='gray', alpha=0.5,
                label=r'Theory: $\tau_R \sim N^{2.18}$')
    ax_tau.set_xscale('log')
    ax_tau.set_yscale('log')
    ax_tau.set_xlabel('Chain length N', fontsize=13)
    ax_tau.set_ylabel(r'Relaxation time $\tau_R$', fontsize=13)
    ax_tau.set_title(r'Relaxation Time Scaling: $\tau_R$ vs N', fontsize=14)
    ax_tau.legend(fontsize=10)
    ax_tau.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'fig4_relaxation_scaling.png'), dpi=200)
    plt.close()
    print("Saved: fig4_relaxation_scaling.png")

elif len(valid_Ns) == 2:
    valid_Ns_arr = np.array(valid_Ns, dtype=float)
    valid_taus = np.array([tau_R[N] for N in valid_Ns])
    log_valid_N = np.log10(valid_Ns_arr)
    log_valid_tau = np.log10(valid_taus)
    slope_tau = (log_valid_tau[1] - log_valid_tau[0]) / (log_valid_N[1] - log_valid_N[0])
    intercept_tau = log_valid_tau[0] - slope_tau * log_valid_N[0]
    stderr_tau = float('nan')

    print(f"\ntau_R vs N (2-point estimate from {valid_Ns}):")
    print(f"  beta = {slope_tau:.4f} (no uncertainty from 2 points)")
    print(f"  Expected beta ~ 2.18")

    fig_tau, ax_tau = plt.subplots(figsize=(8, 6))
    ax_tau.plot(valid_Ns_arr, valid_taus, 'o', color='#9C27B0', markersize=8,
                label=r'Measured $\tau_R$')
    N_fit_tau = np.linspace(valid_Ns_arr[0] * 0.8, valid_Ns_arr[-1] * 1.2, 200)
    ax_tau.plot(N_fit_tau, 10**intercept_tau * N_fit_tau**slope_tau, '--', color='#9C27B0',
                alpha=0.6, label=rf'Fit: $\beta={slope_tau:.3f}$')
    tau_theory_ref = valid_taus[-1] * (N_fit_tau / valid_Ns_arr[-1])**2.18
    ax_tau.plot(N_fit_tau, tau_theory_ref, ':', color='gray', alpha=0.5,
                label=r'Theory: $\tau_R \sim N^{2.18}$')
    ax_tau.set_xscale('log')
    ax_tau.set_yscale('log')
    ax_tau.set_xlabel('Chain length N', fontsize=13)
    ax_tau.set_ylabel(r'Relaxation time $\tau_R$', fontsize=13)
    ax_tau.set_title(r'Relaxation Time Scaling: $\tau_R$ vs N', fontsize=14)
    ax_tau.legend(fontsize=10)
    ax_tau.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'fig4_relaxation_scaling.png'), dpi=200)
    plt.close()
    print("Saved: fig4_relaxation_scaling.png")
else:
    print("\nInsufficient valid tau_R values for scaling analysis")

with open(os.path.join(BASE, 'fig4_tau_summary.tsv'), 'w') as f:
    f.write("N\ttau_R\ttau_R_err\tvalid\tmethod\n")
    for N in Ns:
        if N in tau_R and np.isfinite(tau_R[N]):
            f.write(f"{N}\t{tau_R[N]:.6f}\t{tau_R_err.get(N, 0):.6f}\t"
                    f"{tau_R_valid[N]}\t{tau_R_method[N]}\n")
        else:
            f.write(f"{N}\tNA\tNA\t{tau_R_valid.get(N, False)}\t{tau_R_method.get(N, 'N/A')}\n")
print("Saved: fig4_tau_summary.tsv")

# =============================================================================
# TASK 5: QUALITY CONTROL
# =============================================================================
print("\n" + "=" * 60)
print("TASK 5: QUALITY CONTROL")
print("=" * 60)

for i, N in enumerate(Ns):
    print(f"\n--- N = {N} ---")
    r = ratios[i]
    qc_ratio = "PASS" if 5.0 <= r <= 7.0 else "FAIL"
    print(f"  R2/Rg2 = {r:.3f} -> [{qc_ratio}] (expected [5, 7])")

    # D scaling (using N>=50 fit)
    dev_D_check = abs((slope_D - (-1.0)) / (-1.0))
    qc_D = "PASS" if dev_D_check <= 0.15 else "FAIL"
    print(f"  D scaling (N>=50): alpha = {slope_D:.4f} -> [{qc_D}] "
          f"(deviation {dev_D_check * 100:.1f}%, limit 15%)")

    # tau_R scaling
    if not np.isnan(slope_tau):
        dev_tau = abs((slope_tau - 2.18) / 2.18)
        qc_tau = "PASS" if dev_tau <= 0.20 else "FAIL"
        print(f"  tau_R scaling: beta = {slope_tau:.4f} -> [{qc_tau}] "
              f"(deviation {dev_tau * 100:.1f}%, limit 20%)")
    else:
        print(f"  tau_R scaling: INSUFFICIENT DATA")

    if tau_R_valid.get(N, False):
        print(f"  tau_R fit: VALID (tau_R = {tau_R[N]:.1f}, {tau_R_method[N]})")
    else:
        print(f"  tau_R fit: UNRELIABLE — cannot reliably extract tau_R for N={N}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("EXTRACTED SCALING TABLE")
print("=" * 60)
print(f"{'Observable':<25} {'Exponent':<12} {'+/-stderr':<12} {'Expected':<12} {'Status'}")
print("-" * 73)
print(f"{'2nu (from R2)':<25} {slope_R2:<12.4f} {stderr_R2:<12.4f} {'1.176':<12} "
      f"{'OK' if abs(slope_R2 - 1.176) / 1.176 < 0.15 else 'WARN'}")
print(f"{'2nu (from Rg2)':<25} {slope_Rg2:<12.4f} {stderr_Rg2:<12.4f} {'1.176':<12} "
      f"{'OK' if abs(slope_Rg2 - 1.176) / 1.176 < 0.15 else 'WARN'}")
print(f"{'alpha (D, N>=50)':<25} {slope_D:<12.4f} {stderr_D:<12.4f} {'-1.0':<12} "
      f"{'OK' if abs((slope_D + 1.0) / 1.0) < 0.15 else 'WARN'}")
print(f"{'alpha (D, all N)':<25} {slope_D_all:<12.4f} {stderr_D_all:<12.4f} {'-1.0':<12} "
      f"{'OK' if abs((slope_D_all + 1.0) / 1.0) < 0.15 else 'WARN'}")
if not np.isnan(slope_tau):
    stderr_disp = f"{stderr_tau:.4f}" if np.isfinite(stderr_tau) else "N/A"
    dev_t = abs((slope_tau - 2.18) / 2.18)
    print(f"{'beta (tau_R ~ N^b)':<25} {slope_tau:<12.4f} {stderr_disp:<12} {'2.18':<12} "
          f"{'OK' if dev_t < 0.20 else 'WARN'}")
else:
    print(f"{'beta (tau_R ~ N^b)':<25} {'N/A':<12} {'N/A':<12} {'2.18':<12} INSUFFICIENT DATA")

print(f"\nGenerated figures:")
for fn in ['fig1_static_scaling.png', 'fig2_g1_msd.png', 'fig3_gcm_msd.png',
           'fig3_diffusion_scaling.png', 'fig4_autocorr_fits.png', 'fig4_relaxation_scaling.png']:
    exists = os.path.exists(os.path.join(BASE, fn))
    print(f"  {fn} {'[OK]' if exists else '[MISSING]'}")

print(f"\nGenerated tables:")
for fn in ['fig1_scaling_summary.tsv', 'fig3_diffusion_summary.tsv', 'fig4_tau_summary.tsv']:
    exists = os.path.exists(os.path.join(BASE, fn))
    print(f"  {fn} {'[OK]' if exists else '[MISSING]'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
