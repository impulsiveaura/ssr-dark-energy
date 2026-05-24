"""
geff_camb_check.py
==================
Computes linear growth factor and matter power spectrum for SSR vs ΛCDM
using oscillatory G_eff(z) modification.

Corresponds to Figure 7 of:
  Rahman, S.S. (2026). Sovereign Scaling Resonance (SSR): an Oscillatory
  Dark Energy Extension of ΛCDM with Cosmological Constraints and
  Laboratory Falsifiability. doi:10.5281/zenodo.20350090

Requirements: camb, numpy, scipy, matplotlib
Usage: python3 geff_camb_check.py
"""

import numpy as np
import camb
from camb import model
from scipy.integrate import odeint, cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── SSR best-fit parameters ───────────────────────────────────────────────────
H0    = 70.76
ombh2 = 0.02242
omch2 = (0.2715 - 0.049) * (H0/100)**2
Om0   = 0.2715
A_ssr = 0.031
omega = 5.40
phi   = 3.38
fM    = 0.503   # in M_pl units

def Geff_ratio(z):
    """G_eff(z)/G for SSR: eq. (8) of the paper."""
    return 1.0 - 3*A_ssr*fM**2 * np.sin(omega*np.log(1+z) + phi)

# ── ΛCDM baseline power spectrum via CAMB ────────────────────────────────────
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
pars.set_matter_power(redshifts=[0.0, 0.5, 1.0, 1.5, 2.0], kmax=2.0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z_arr, pk_lcdm = results.get_matter_power_spectrum(
    minkh=1e-4, maxkh=1.0, npoints=200)
sigma8_lcdm = results.get_sigma8_0()

# ── Growth ODE with G_eff(z) ─────────────────────────────────────────────────
def E2(a):
    return Om0/a**3 + (1-Om0)

def Om_a(a):
    return Om0/a**3/E2(a)

def dlnH2(a):
    return -3*Om0/a**3/E2(a)

def growth_ode(y, lna, use_Geff=True):
    a = np.exp(lna)
    z = 1/a - 1
    f = y[0]
    Geff = Geff_ratio(z) if use_Geff else 1.0
    return [-f**2 - (2 + 0.5*dlnH2(a))*f + 1.5*Om_a(a)*Geff]

lna = np.linspace(-7, 0, 3000)
f_ic = [Om0**0.55]

sol_SSR  = odeint(growth_ode, f_ic, lna, args=(True,))[:,0]
sol_LCDM = odeint(growth_ode, f_ic, lna, args=(False,))[:,0]

lnD_SSR  = cumulative_trapezoid(sol_SSR,  lna, initial=0)
lnD_LCDM = cumulative_trapezoid(sol_LCDM, lna, initial=0)
D_SSR  = np.exp(lnD_SSR);  D_SSR  /= D_SSR[-1]
D_LCDM = np.exp(lnD_LCDM); D_LCDM /= D_LCDM[-1]

a_arr = np.exp(lna)
D_SSR_i  = interp1d(a_arr, D_SSR,  bounds_error=False, fill_value='extrapolate')
D_LCDM_i = interp1d(a_arr, D_LCDM, bounds_error=False, fill_value='extrapolate')

# ── Print results ─────────────────────────────────────────────────────────────
print('Growth factor ratio D_SSR/D_LCDM:')
for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
    ratio = D_SSR_i(1/(1+z)) / D_LCDM_i(1/(1+z))
    print(f'  z={z:.1f}: {ratio:.6f} ({(ratio-1)*100:+.4f}%)')

S8_SSR  = sigma8_lcdm * float(D_SSR_i(1.0))  * np.sqrt(Om0/0.3)
S8_LCDM = sigma8_lcdm * float(D_LCDM_i(1.0)) * np.sqrt(0.270/0.3)
print(f'\nS8_SSR  = {S8_SSR:.4f}')
print(f'S8_LCDM = {S8_LCDM:.4f}')
print(f'KiDS-1000 target: 0.766, DES Y3 target: 0.776')

# ── Figure 7 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#e05c5c', '#e08c5c', '#5c8ce0']
for iz, (z, col) in enumerate(zip([0.0, 0.5, 1.0], colors)):
    D_ratio = D_SSR_i(1/(1+z)) / D_LCDM_i(1/(1+z))
    pk_z = pk_lcdm[iz]
    axes[0].loglog(kh, pk_z,            ls='--', color=col, alpha=0.7, label=f'ΛCDM z={z}')
    axes[0].loglog(kh, pk_z*D_ratio**2, ls='-',  color=col, alpha=0.9, label=f'SSR z={z}')

axes[0].set_xlabel('k [h/Mpc]');  axes[0].set_ylabel('P(k) [(Mpc/h)³]')
axes[0].set_title('Matter Power Spectrum: SSR vs ΛCDM')
axes[0].legend(fontsize=7);  axes[0].grid(True, alpha=0.3)

z_plot = np.linspace(0, 2.5, 500)
axes[1].plot(z_plot, Geff_ratio(z_plot), color='#60a5fa', lw=2, label='SSR G_eff(z)')
axes[1].axhline(1.0, color='orange', ls='--', lw=1.5, label='ΛCDM (G_eff = G)')
axes[1].fill_between(z_plot, 1.0, Geff_ratio(z_plot), alpha=0.15, color='#60a5fa')
axes[1].set_xlabel('Redshift z');  axes[1].set_ylabel('G_eff(z)/G')
axes[1].set_title('Oscillatory G_eff(z) in SSR')
axes[1].legend();  axes[1].grid(True, alpha=0.3);  axes[1].set_xlim(0, 2.5)

fig.tight_layout()
fig.savefig('SSR_Geff_power_spectrum.png', dpi=150, bbox_inches='tight')
print('\nFigure saved: SSR_Geff_power_spectrum.png')
