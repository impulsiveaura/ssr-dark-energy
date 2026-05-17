#!/usr/bin/env python3
"""
generate_figures.py
Generates Fig. 1 for the SSR PDU paper:
  Left panel:  H(z) for SSR best-fit vs LCDM
  Right panel: w(z) for SSR best-fit with 68% credible band
Run after ssr_v4 chains have converged.
Output: ~/ssr_run/figures/SSR_fig1_Hz_wz.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob, os, sys
sys.path.insert(0, os.path.expanduser('~/ssr_run'))

# ── Load chains ──────────────────────────────────────────────────────────────
ssr_files  = sorted(glob.glob(os.path.expanduser('~/ssr_run/chains/ssr_v4.*.txt')))
lcdm_files = sorted(glob.glob(os.path.expanduser('~/ssr_run/chains/lcdm_v2.*.txt')))

if not ssr_files:
    # Fall back to v3 single chain if v4 not ready
    ssr_files = [os.path.expanduser('~/ssr_run/chains/ssr_v3.1.txt')]
    print("Using ssr_v3 chain (ssr_v4 not found)")

ssr  = np.vstack([np.loadtxt(f, comments='#') for f in ssr_files])
lcdm = np.loadtxt(lcdm_files[0], comments='#') if lcdm_files else None

# Column order for SSR chain
# weight, minuslogpost, H0, Omega_m, ssr_A, ssr_Lambda4, ssr_beta,
# ssr_f_Xi, ssr_lambda, ssr_omega, ssr_phi, chi2__SN, ...
cols = ['weight','minuslogpost','H0','Omega_m','ssr_A','ssr_Lambda4',
        'ssr_beta','ssr_f_Xi','ssr_lambda','ssr_omega','ssr_phi']

def col(name): return cols.index(name)

# Best-fit and median SSR parameters
bf = ssr[np.argmin(ssr[:,1])]
H0_bf   = bf[col('H0')]
Om_bf   = bf[col('Omega_m')]
A_bf    = bf[col('ssr_A')]
om_bf   = bf[col('ssr_omega')]
phi_bf  = bf[col('ssr_phi')]

H0_med  = np.median(ssr[:, col('H0')])
A_med   = np.median(ssr[:, col('ssr_A')])
om_med  = np.median(ssr[:, col('ssr_omega')])
phi_med = np.median(ssr[:, col('ssr_phi')])

# LCDM best-fit
if lcdm is not None:
    lcdm_cols = ['weight','minuslogpost','H0','Omega_m']
    lcdm_cols = lcdm_cols[:lcdm.shape[1]]
    H0_lcdm = np.median(lcdm[:, 2])
    Om_lcdm = np.median(lcdm[:, 3])
else:
    H0_lcdm, Om_lcdm = 71.92, 0.2706

# ── Cosmology functions ───────────────────────────────────────────────────────
_OMEGA_R0_H2 = 2.469e-5

def H_LCDM(z, H0, Om):
    h = H0/100.; Or = _OMEGA_R0_H2/h**2; OL = 1-Om-Or
    return H0 * np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL)

def w_SSR(z, A, omega, phi):
    return -1 + A * np.sin(omega * np.log(1+z) + phi)

def H_SSR_approx(z, H0, Om, A, omega, phi):
    """Approximate H(z) for SSR using background integral."""
    h = H0/100.; Or = _OMEGA_R0_H2/h**2; OL = 1-Om-Or
    # Integrate dark energy density: rho_DE propto exp(3 int (1+w)dlna)
    # For small A, approximate as LCDM + oscillatory correction
    zz = np.atleast_1d(z)
    correction = np.zeros_like(zz, dtype=float)
    for i, zi in enumerate(zz):
        if zi > 0:
            lna = np.linspace(0, np.log(1+zi), 500)
            z_int = np.exp(lna) - 1
            w_int = w_SSR(z_int, A, omega, phi)
            integrand = 3*(1 + w_int)
            correction[i] = np.trapz(integrand, lna)
    rho_DE = OL * np.exp(correction)
    return H0 * np.sqrt(Om*(1+zz)**3 + Or*(1+zz)**4 + rho_DE)

# ── Redshift arrays ───────────────────────────────────────────────────────────
z = np.linspace(0, 2.5, 300)

H_lcdm = H_LCDM(z, H0_lcdm, Om_lcdm)
H_ssr   = H_SSR_approx(z, H0_bf, Om_bf, A_bf, om_bf, phi_bf)
w_ssr   = w_SSR(z, A_med, om_med, phi_med)

# 68% band on w(z) from chain
n_samples = min(2000, len(ssr))
idx = np.random.choice(len(ssr), n_samples, replace=False)
w_samples = np.array([w_SSR(z, ssr[i,col('ssr_A')], ssr[i,col('ssr_omega')], ssr[i,col('ssr_phi')]) for i in idx])
w_lo = np.percentile(w_samples, 16, axis=0)
w_hi = np.percentile(w_samples, 84, axis=0)

# ── DESI BAO data points (approximate) ───────────────────────────────────────
# DH/rd values converted to H(z) approximations for display
bao_z   = np.array([0.295, 0.510, 0.706, 0.934, 1.317, 1.491, 2.330])
bao_H   = np.array([69.0,  79.8,  89.4, 100.4, 118.3, 127.0, 228.0])
bao_err = np.array([2.1,   2.5,   3.0,   3.5,   5.0,   6.0,  14.0])

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel 1: H(z)
ax1.plot(z, H_lcdm, color='#E07B39', lw=2.0, label=r'$\Lambda$CDM  ($H_0=%.1f$)' % H0_lcdm, zorder=3)
ax1.plot(z, H_ssr,  color='#3B7DD8', lw=2.0, label=r'SSR best-fit ($H_0=%.1f$)' % H0_bf,   zorder=3)
ax1.errorbar(bao_z, bao_H, yerr=bao_err, fmt='o', color='#2D2D2D',
             markersize=5, capsize=3, lw=1.2, label='DESI 2024 BAO', zorder=4)
ax1.set_xlabel(r'Redshift $z$', fontsize=13)
ax1.set_ylabel(r'$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=13)
ax1.set_title(r'Hubble parameter $H(z)$', fontsize=13)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.set_xlim(0, 2.5)
ax1.set_ylim(60, 260)
ax1.tick_params(labelsize=11)
ax1.grid(True, alpha=0.25, lw=0.5)

# Panel 2: w(z)
ax2.axhline(-1, color='#E07B39', lw=1.8, ls='--', label=r'$\Lambda$CDM: $w=-1$', zorder=2)
ax2.fill_between(z, w_lo, w_hi, color='#3B7DD8', alpha=0.25, label='SSR 68% CI', zorder=3)
ax2.plot(z, w_ssr, color='#3B7DD8', lw=2.0, label='SSR median', zorder=4)
ax2.axhline(0, color='gray', lw=0.6, ls=':', alpha=0.5)
ax2.set_xlabel(r'Redshift $z$', fontsize=13)
ax2.set_ylabel(r'Equation of state $w(z)$', fontsize=13)
ax2.set_title(r'Dark energy equation of state $w(z)$', fontsize=13)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.set_xlim(0, 2.5)
ax2.set_ylim(-1.25, -0.75)
ax2.tick_params(labelsize=11)
ax2.grid(True, alpha=0.25, lw=0.5)

# Annotation
fig.text(0.5, 0.01,
         r'SSR: $w(z) \approx -1 + A\sin[\omega\ln(1+z)+\phi]$   |   '
         r'$A=%.3f,\ \omega=%.2f,\ \phi=%.2f$' % (A_med, om_med, phi_med),
         ha='center', fontsize=10, style='italic', color='#444444')

os.makedirs(os.path.expanduser('~/ssr_run/figures'), exist_ok=True)
outpath = os.path.expanduser('~/ssr_run/figures/SSR_fig1_Hz_wz.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
plt.close()
