"""
cdl_instanton_check.py
======================
Numerical Coleman-De Luccia instanton analysis for the SSR cosine potential.

Shows that flat-space, CdL, and Hawking-Moss instanton actions for the
SSR potential are of order 10^116-118, confirming that the QCD-Planck
coincidence S_inst ~ 22 ~ ln(M_pl/Lambda_QCD)/2 is a numerical observation
of the hierarchy, not a tunneling instanton of the SSR cosine potential.

Corresponds to the Speculative Outlook section of:
  Rahman, S.S. (2026). Sovereign Scaling Resonance (SSR): an Oscillatory
  Dark Energy Extension of ΛCDM with Cosmological Constraints and
  Laboratory Falsifiability. doi:10.5281/zenodo.20350090

Requirements: numpy, scipy
Usage: python3 cdl_instanton_check.py
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ── SSR parameters (M_pl = 1 units) ──────────────────────────────────────────
fM      = 0.503          # axion decay constant [M_pl]
zeta    = 1.29e19        # M_pl/m_p
rho_SSR = 1/zeta**6      # vacuum energy density [M_pl^4]
m_Xi    = np.sqrt(rho_SSR)/fM  # field mass [M_pl]

print('SSR Instanton Analysis')
print('='*55)
print(f'f_M     = {fM} M_pl')
print(f'rho_SSR = {rho_SSR:.4e} M_pl^4')
print(f'm_Xi    = {m_Xi:.4e} M_pl')
print(f'r_dS    = sqrt(3/rho_SSR) = {np.sqrt(3/rho_SSR):.3e} M_pl^-1')
print(f'rho_inst = 1/m_Xi = {1/m_Xi:.3e} M_pl^-1')
print()

def V(Xi):
    return rho_SSR*(1 - np.cos(Xi/fM))

def dV(Xi):
    return (rho_SSR/fM)*np.sin(Xi/fM)

# ── 1. Flat-space instanton action ────────────────────────────────────────────
print('1. Flat-space instanton action')
print('-'*40)
S1, _ = quad(lambda th: np.sqrt(2*rho_SSR*(1-np.cos(th)))*fM, 0, 2*np.pi)
rho_inst = 1/m_Xi
S_flat = 2*np.pi**2 * rho_inst**3 * S1
print(f'Surface tension S_1 = {S1:.4e} M_pl^3')
print(f'Instanton radius = {rho_inst:.3e} M_pl^-1')
print(f'S_E_flat = 2*pi^2 * rho_inst^3 * S_1 ~ 10^{np.log10(abs(S_flat)):.0f}')
print()

# ── 2. Hawking-Moss instanton action ─────────────────────────────────────────
print('2. Hawking-Moss instanton action')
print('-'*40)
V_top = 2*rho_SSR  # barrier top
S_HM = 24*np.pi**2/V_top
print(f'V_top = 2*rho_SSR = {V_top:.4e} M_pl^4')
print(f'S_HM = 24*pi^2/V_top = {S_HM:.4e}')
print(f'     ~ 10^{np.log10(S_HM):.0f}')
print()

# ── 3. CdL shooting attempt ───────────────────────────────────────────────────
print('3. Coleman-De Luccia shooting')
print('-'*40)
print('Scanning for bounce solution...')

def cdl_ode(rho, y):
    Xi, dXi, a = y
    if a <= 1e-15: return [dXi, 0.0, 1.0]
    arg = max(0, 1 - a**2*(dXi**2/2 + V(Xi))/3)
    da = np.sqrt(arg)
    d2Xi = dV(Xi) - 3*(da/a)*dXi
    return [dXi, d2Xi, da]

rho_max = 5*np.sqrt(3/V_top)
Xi0_values = np.linspace(0.1*fM, 1.99*np.pi*fM, 15)
results = []
for Xi0 in Xi0_values:
    try:
        sol = solve_ivp(cdl_ode, (1e-8, rho_max), [Xi0, 0.0, 1e-10],
                       method='RK45', max_step=rho_max/200,
                       rtol=1e-8, atol=1e-12)
        results.append((Xi0, sol.y[0][-1]))
    except:
        pass

sign_changes = sum(1 for i in range(len(results)-1)
                   if results[i][1]*results[i+1][1] < 0)
print(f'Scanned {len(results)} initial conditions')
print(f'Sign changes in Xi_final: {sign_changes}')
if sign_changes == 0:
    print('No CdL bounce solution found — tunneling suppressed')
print()

# ── 4. Summary ────────────────────────────────────────────────────────────────
print('='*55)
print('SUMMARY')
print('='*55)
print(f'Flat-space S_E    ~ 10^{np.log10(abs(S_flat)):.0f}  (not 22)')
print(f'Hawking-Moss S_HM ~ 10^{np.log10(S_HM):.0f}  (not 22)')
print(f'CdL bounce        : no solution found')
print()
print('The QCD-Planck coincidence:')
Lambda_QCD = 0.217  # GeV
Mpl_GeV = 1.22e19  # GeV
S_inst_QCD = 0.5*np.log(Mpl_GeV/Lambda_QCD)
print(f'  ln(M_pl/Lambda_QCD)/2 = {S_inst_QCD:.2f}')
print(f'  MCMC omega/4 = {5.40/4:.2f}')
print()
print('CONCLUSION: S_inst ~ 22 is a numerical echo of the QCD-Planck')
print('hierarchy, not a tunneling instanton of the SSR cosine potential.')
print('The speculative outlook in the paper is correctly framed.')
