"""
SSR Numerical Module v1.0
=========================
Complete numerical implementation of the Sovereign Scaling Resonance
framework for the companion paper.

Computes:
  - Background: H(z), D_A(z), D_L(z) from coupled {H, Xi, chi} ODE system
  - Perturbations: growth factor f(z), fsigma8(z), sigma8(z)
  - Power spectrum: P(k,z) with scale-dependent G_eff suppression
  - Chi-squared comparison against DESI DR1, KiDS-1000, Planck lensing data

Parameters:
  theta = {lambda, beta, Lambda4, f_Xi, n, A, omega, phi}
  LCDM baseline: {H0, Omega_m, Omega_b, Omega_r, sigma8_0, n_s}

Author: Sharaf Samiur Rahman
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONSTANTS AND FIDUCIAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Physical constants
c_km = 2.998e5        # speed of light in km/s
H0_fid = 67.4         # km/s/Mpc (Planck 2018)
Omega_m0 = 0.315      # matter density (Planck 2018)
Omega_b0 = 0.049      # baryon density
Omega_r0 = 9.0e-5     # radiation density
Omega_L0 = 1.0 - Omega_m0 - Omega_r0
sigma8_LCDM = 0.811   # Planck 2018
n_s = 0.965           # spectral index

# H0 in units of M_P (for internal SSR calculations)
H0_MP = 1.18e-61

# SSR fiducial parameters (benchmark model)
SSR_FIDUCIAL = {
    'lambda': 0.5,      # coherence coupling
    'beta':   0.15,     # non-minimal curvature coupling
    'Lambda4': 1e-4,    # potential scale (H0^2 units)
    'f_Xi':   0.1,      # field decay constant (M_P units)
    'n':      6.0,      # hierarchy suppression exponent (paper: p_SSR = M_p^4/zeta^6)
    'A':      0.015,    # expansion modulation amplitude
    'omega':  2.0,      # oscillation frequency in ln(a)
    'phi':    0.3,      # oscillation phase
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BACKGROUND COSMOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def _get_cosmo(params=None):
    """Return (H0, Omega_m, Omega_r, Omega_L) from params dict or module globals."""
    if params is not None:
        H0   = params.get('H0',      H0_fid)
        Om   = params.get('Omega_m', Omega_m0)
        Or   = params.get('Omega_r', Omega_r0)
    else:
        H0, Om, Or = H0_fid, Omega_m0, Omega_r0
    OL = 1.0 - Om - Or
    return H0, Om, Or, OL

def H_LCDM(a, H0=None, params=None):
    """LCDM Hubble rate H(a) in km/s/Mpc."""
    H0_, Om, Or, OL = _get_cosmo(params)
    if H0 is None:
        H0 = H0_
    return H0 * np.sqrt(Om * a**-3 + Or * a**-4 + OL)

def dH_dlna_LCDM(a, H0=None, params=None):
    """d(H)/d(ln a) for LCDM."""
    H0_, Om, Or, OL = _get_cosmo(params)
    if H0 is None:
        H0 = H0_
    H = H0 * np.sqrt(Om * a**-3 + Or * a**-4 + OL)
    num = H0**2 * (-1.5 * Om * a**-3 - 2.0 * Or * a**-4)
    return num / H

def V_Xi(Xi, Lambda4, f_Xi):
    """Cosine potential V(Xi) = Lambda4 * [1 - cos(Xi/f_Xi)]"""
    return Lambda4 * (1.0 - np.cos(Xi / f_Xi))

def dV_dXi(Xi, Lambda4, f_Xi):
    return Lambda4 / f_Xi * np.sin(Xi / f_Xi)

def solve_background(params, a_arr=None):
    """
    Solve the SSR background equations for {Xi(a), dXi/dlna(a)}.
    Uses LCDM for H(a) (valid for lambda << 1 or as leading order).
    Returns interpolators for Xi(a) and dXi/dlna(a).
    """
    lam   = params['lambda']
    beta  = params['beta']
    L4    = params['Lambda4']
    f_Xi  = params['f_Xi']

    if a_arr is None:
        a_arr = np.logspace(-4, 0, 500)

    lna_arr = np.log(a_arr)

    def rhs(lna, y):
        a   = np.exp(lna)
        Xi  = y[0]
        dXi = y[1]   # dXi/d(ln a)

        H    = H_LCDM(a, params=params) / params.get('H0', H0_fid)   # dimensionless H/H0
        dH   = dH_dlna_LCDM(a, params=params) / params.get('H0', H0_fid)
        eps_H = dH / H if H > 0 else 0.0

        friction      = 3.0 + eps_H
        pot_term      = dV_dXi(Xi, L4, f_Xi) / (H**2 + 1e-300)
        R_over_H2     = 6.0 * (eps_H + 2.0)
        coh_mass      = 2.0 * lam * beta * R_over_H2

        dXi_dlna   = dXi
        d2Xi_dlna2 = -(friction * dXi + pot_term + coh_mass * Xi) / (1.0 + lam)
        return [dXi_dlna, d2Xi_dlna2]

    y0 = [1e-3, 0.0]
    sol = solve_ivp(rhs, [lna_arr[0], lna_arr[-1]], y0,
                    t_eval=lna_arr, method='DOP853',
                    rtol=1e-9, atol=1e-11, max_step=0.05)

    Xi_interp   = CubicSpline(a_arr, sol.y[0])
    dXi_interp  = CubicSpline(a_arr, sol.y[1])
    return Xi_interp, dXi_interp

def H_SSR(a, params, Xi_interp, dXi_interp):
    """
    SSR Hubble rate. The coherence sector energy density rho_coh ~ -3*lambda*beta*H^2*Xi^2
    gives a fractional correction to H of ~7.5e-8 at fiducial parameters (Xi~1e-3 M_P),
    which is 5 orders of magnitude below DESI BAO precision. It is negligible.
    The only non-negligible background deviation from LCDM is the oscillatory term.
    See ssr_flrw_check.py for the full derivation.
    """
    H_L = H_LCDM(a, params=params) / params.get('H0', H0_fid)

    # SSR oscillatory modulation (eq 7.1) - only non-negligible background term
    A   = params['A']
    om  = params['omega']
    phi = params['phi']
    lna = np.log(a)
    DH_mod = A * np.sin(om * lna + phi)

    H_total = params.get('H0', H0_fid) * H_L * (1.0 + DH_mod)
    return H_total

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DISTANCE MEASURES
# ═══════════════════════════════════════════════════════════════════════════════

def comoving_distance(z, params, Xi_interp, dXi_interp):
    """Comoving distance chi(z) in Mpc."""
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        H = H_SSR(a, params, Xi_interp, dXi_interp)
        return c_km / H

    if np.isscalar(z):
        result, _ = quad(integrand, 0, z, limit=200)
        return result
    else:
        return np.array([quad(integrand, 0, zi, limit=200)[0] for zi in z])

def angular_diameter_distance(z, params, Xi_interp, dXi_interp):
    """D_A(z) in Mpc."""
    chi = comoving_distance(z, params, Xi_interp, dXi_interp)
    return chi / (1.0 + z)

def luminosity_distance(z, params, Xi_interp, dXi_interp):
    """D_L(z) in Mpc."""
    chi = comoving_distance(z, params, Xi_interp, dXi_interp)
    return chi * (1.0 + z)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PERTURBATION GROWTH
# ═══════════════════════════════════════════════════════════════════════════════

def G_eff(k_hMpc, a, params, Xi_interp):
    """
    Effective gravitational coupling from SSR eq (8.2)-(8.3).
    G_eff = G * (1 - sigma(k, a))
    sigma = 8*lambda*beta*k^2*Xi^2 / (k^2 + m_eff^2)^2
    k in h/Mpc, converted to M_P units internally.
    """
    lam  = params['lambda']
    beta = params['beta']
    L4   = params['Lambda4']
    f_Xi = params['f_Xi']
    Xi   = Xi_interp(a)

    # Convert k and m_eff to the same physical units [1/Mpc].
    # k [h/Mpc] -> k [1/Mpc] = k_hMpc * h   (h = H0_fid/100)
    # m_eff^2 is in H0^2 units (Lambda4 stored in H0^2 units).
    # m_eff [1/Mpc] = sqrt(m_eff2_H0) * H0/c
    H0_local = params.get('H0', H0_fid)  # use sampled H0 for unit conversion
    h_dimless = H0_local / 100.0
    k_invMpc = k_hMpc * h_dimless

    # Effective mass of Xi in H0^2 units, then convert to 1/Mpc^2
    m_eff2_H0 = L4 / f_Xi**2 * np.cos(Xi / f_Xi)
    m_eff2_H0 = max(m_eff2_H0, 1e-30)   # prevent negative mass squared
    m_eff2_invMpc = m_eff2_H0 * (H0_local / c_km) ** 2

    # Scale-dependent suppression (eq 8.3) — all quantities now in 1/Mpc^2
    denom = (k_invMpc**2 + m_eff2_invMpc)**2
    sigma = 8.0 * lam * beta * k_invMpc**2 * Xi**2 / (denom + 1e-300)
    sigma = np.clip(sigma, 0.0, 0.99)

    return 1.0 - sigma

def growth_ode(lna, y, k_hMpc, params, Xi_interp):
    """
    ODE for linear density contrast delta and its derivative.
    y = [delta, d(delta)/d(ln a)]
    Eq (8.1): delta'' + (2 + eps_H) delta' - 1.5*Omega_m*G_eff/G * delta/a^3 = 0
    """
    a    = np.exp(lna)
    d    = y[0]
    dp   = y[1]   # d(delta)/d(ln a)

    H    = H_LCDM(a, params=params) / params.get('H0', H0_fid)
    dH   = dH_dlna_LCDM(a, params=params) / params.get('H0', H0_fid)
    eps_H = dH / H if H > 0 else 0.0

    Geff = G_eff(k_hMpc, a, params, Xi_interp)

    Om = params.get('Omega_m', Omega_m0)
    # Source term: 1.5 * Omega_m0 * G_eff/G / (a^3 * H^2)
    source = 1.5 * Om * Geff / (a**3 * H**2 + 1e-300)

    dd  = dp
    ddp = -(2.0 + eps_H) * dp + source * d
    return [dd, ddp]

def compute_growth(params, Xi_interp, k_arr_hMpc=None, z_arr=None):
    """
    Compute growth factor D(k,z) and growth rate f(k,z) = d ln D / d ln a.
    Returns fsigma8(z) normalised to Planck sigma8 at z=0 for LCDM.
    """
    if k_arr_hMpc is None:
        k_arr_hMpc = np.logspace(-3, 1, 60)
    if z_arr is None:
        z_arr = np.array([0.0, 0.2, 0.4, 0.57, 0.8, 1.0])

    a_arr_z = 1.0 / (1.0 + z_arr)
    k_rep = 0.1   # h/Mpc representative scale

    lna_i = np.log(1e-4)
    lna_f = 0.0

    # --- LCDM reference: solve with lambda=0, beta=0 ---
    params_lcdm = {**params, 'lambda': 0.0, 'beta': 0.0}
    sol_lcdm = solve_ivp(growth_ode, [lna_i, lna_f], [1e-4, 1e-4],
                         args=(k_rep, params_lcdm, Xi_interp),
                         method='DOP853', rtol=1e-9, atol=1e-11,
                         dense_output=True, max_step=0.05)
    D_lcdm_today = sol_lcdm.y[0, -1]

    # --- SSR solution ---
    sol = solve_ivp(growth_ode, [lna_i, lna_f], [1e-4, 1e-4],
                    args=(k_rep, params, Xi_interp),
                    method='DOP853', rtol=1e-9, atol=1e-11,
                    dense_output=True, max_step=0.05)

    lna_eval = np.clip(np.log(a_arr_z), lna_i, lna_f)
    y_eval = sol.sol(lna_eval)
    y_lcdm_eval = sol_lcdm.sol(lna_eval)

    delta_arr  = y_eval[0]
    ddelta_arr = y_eval[1]
    delta_lcdm = y_lcdm_eval[0]

    # Growth rate f = d ln delta / d ln a
    f_arr = ddelta_arr / (delta_arr + 1e-300)

    # Normalise: D_SSR relative to LCDM today, then scale by sigma8_LCDM
    # D_ratio(z) = D_SSR(z) / D_LCDM(z=0)
    D_ratio = delta_arr / (D_lcdm_today + 1e-300)

    # sigma8(z) calibrated to Planck at z=0 for LCDM; SSR suppresses by D ratio
    D_ssr_today = sol.y[0, -1]
    sigma8_z0_SSR = sigma8_LCDM * (D_ssr_today / (D_lcdm_today + 1e-300))
    sigma8_arr = sigma8_LCDM * D_ratio

    # fsigma8(z) = f(z) * sigma8(z)
    fsigma8_arr = f_arr * sigma8_arr

    return {
        'z':        z_arr,
        'D':        D_ratio,
        'f':        f_arr,
        'sigma8':   sigma8_arr,
        'fsigma8':  fsigma8_arr,
        'sigma8_z0': sigma8_z0_SSR,
    }

def compute_Pk(params, Xi_interp, k_arr_hMpc=None, z=0.0):
    """
    Compute SSR matter power spectrum P(k,z) relative to LCDM.
    P_SSR(k) = P_LCDM(k) * [D_SSR(k,z) / D_LCDM(z)]^2
    where D_SSR is scale-dependent due to G_eff(k).
    Returns ratio P_SSR/P_LCDM and the SSR k* feature scale.
    """
    if k_arr_hMpc is None:
        k_arr_hMpc = np.logspace(-3, 1, 80)

    a_z = 1.0 / (1.0 + z)
    lna_i = np.log(1e-4)
    lna_f = np.log(a_z)

    D_ratio = np.ones(len(k_arr_hMpc))

    # LCDM reference growth at this z
    sol_ref = solve_ivp(growth_ode, [lna_i, lna_f], [1e-4, 1e-4],
                        args=(0.1, {**params, 'lambda': 0.0, 'beta': 0.0},
                              Xi_interp),
                        method='DOP853', rtol=1e-8, atol=1e-10,
                        dense_output=True)
    D_lcdm = sol_ref.y[0, -1]

    for i, k in enumerate(k_arr_hMpc):
        sol_k = solve_ivp(growth_ode, [lna_i, lna_f], [1e-4, 1e-4],
                          args=(k, params, Xi_interp),
                          method='DOP853', rtol=1e-8, atol=1e-10,
                          dense_output=True)
        D_ssr = sol_k.y[0, -1]
        D_ratio[i] = (D_ssr / (D_lcdm + 1e-300))**2

    # Find k* (scale of maximum suppression)
    suppress = 1.0 - D_ratio
    k_star_idx = np.argmax(suppress)
    k_star = k_arr_hMpc[k_star_idx]

    return {
        'k': k_arr_hMpc,
        'Pk_ratio': D_ratio,
        'suppression': suppress,
        'k_star': k_star,
        'max_suppression': suppress[k_star_idx],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: OBSERVATIONAL DATA (published values with uncertainties)
# ═══════════════════════════════════════════════════════════════════════════════

# DESI DR1 BAO: D_H/r_d and D_M/r_d (Adame et al 2024, arXiv:2404.03002)
# r_d ~ 147.09 Mpc (Planck 2018 sound horizon)
r_d = 147.09  # Mpc — Planck 2018 fiducial; used as fixed approximation.
# NOTE: r_d depends on Omega_m*h^2 and Omega_b*h^2. Fixing it while varying
# H0 and Omega_m introduces a ~2% systematic in chi2_BAO. A future version
# should use the Eisenstein & Hu (1998) fitting formula:
#   r_d ~ 147.05 * (Omega_m*h^2/0.1431)^{-0.255} * (Omega_b*h^2/0.02226)^{-0.128}

DESI_BAO = {
    'z':      np.array([0.30, 0.51, 0.71, 0.93, 1.32, 2.33]),
    'DH_rd':  np.array([22.23, 20.98, 19.51, 17.90, 16.44, 8.52]),
    'DH_rd_err': np.array([0.55, 0.61, 0.60, 0.51, 0.59, 0.17]),
    'DM_rd':  np.array([7.93, 13.62, 16.85, 21.71, 27.79, 39.71]),
    'DM_rd_err': np.array([0.15, 0.42, 0.61, 0.74, 0.69, 0.94]),
}

# KiDS-1000: S8 = sigma8 * sqrt(Omega_m/0.3)
# Heymans et al 2021: S8 = 0.766 +0.020/-0.014
KIDS_S8 = {'S8': 0.766, 'S8_err_up': 0.020, 'S8_err_dn': 0.014}

# DES Y3: S8 = 0.776 +/- 0.017 (Abbott et al 2022)
DES_S8 = {'S8': 0.776, 'S8_err': 0.017}

# Planck 2018 CMB: S8 = 0.832 +/- 0.013
PLANCK_S8 = {'S8': 0.832, 'S8_err': 0.013}

# BOSS RSD: fsigma8 measurements (Alam et al 2017)
BOSS_RSD = {
    'z':        np.array([0.38, 0.51, 0.61]),
    'fsigma8':  np.array([0.497, 0.458, 0.436]),
    'err':      np.array([0.045, 0.038, 0.034]),
}

# DESI DR1 RSD (preliminary, Gil-Marin et al 2024)
DESI_RSD = {
    'z':        np.array([0.51, 0.71, 0.93]),
    'fsigma8':  np.array([0.455, 0.420, 0.408]),
    'err':      np.array([0.032, 0.029, 0.031]),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CHI-SQUARED COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def chi2_BAO(params, Xi_interp, dXi_interp):
    """Chi-squared for DESI DR1 BAO distances."""
    z_data = DESI_BAO['z']
    chi2 = 0.0

    for i, z in enumerate(z_data):
        a = 1.0 / (1.0 + z)
        H_z = H_SSR(a, params, Xi_interp, dXi_interp)
        chi_z = comoving_distance(z, params, Xi_interp, dXi_interp)

        DH_rd_model = (c_km / H_z) / r_d
        DM_rd_model = chi_z / r_d

        chi2 += ((DH_rd_model - DESI_BAO['DH_rd'][i]) /
                  DESI_BAO['DH_rd_err'][i])**2
        chi2 += ((DM_rd_model - DESI_BAO['DM_rd'][i]) /
                  DESI_BAO['DM_rd_err'][i])**2

    return chi2

def chi2_RSD(params, Xi_interp, growth_results):
    """Chi-squared for BOSS + DESI fsigma8."""
    chi2 = 0.0
    z_gr = growth_results['z']
    fs8_gr = growth_results['fsigma8']
    # sort ascending
    sort_idx = np.argsort(z_gr)
    fsigma8_interp = CubicSpline(z_gr[sort_idx], fs8_gr[sort_idx])

    for dataset in [BOSS_RSD, DESI_RSD]:
        for i, z in enumerate(dataset['z']):
            if z <= z_gr.max():
                fs8_model = float(fsigma8_interp(z))
                chi2 += ((fs8_model - dataset['fsigma8'][i]) /
                          dataset['err'][i])**2
    return chi2

def compute_S8(params, Xi_interp, growth_results):
    """S8 = sigma8(z=0) * sqrt(Omega_m/0.3) for SSR."""
    Om = params.get('Omega_m', Omega_m0)
    sigma8_z0 = growth_results.get('sigma8_z0',
                    growth_results['sigma8'][np.argmin(np.abs(growth_results['z']))])
    return sigma8_z0 * np.sqrt(Om / 0.3)

def chi2_S8(params, Xi_interp, growth_results):
    """Combined S8 chi-squared from KiDS + DES."""
    S8_model = compute_S8(params, Xi_interp, growth_results)

    # KiDS-1000 (asymmetric error: use upper for model > data, lower otherwise)
    dS8_kids = S8_model - KIDS_S8['S8']
    err_kids = KIDS_S8['S8_err_up'] if dS8_kids > 0 else KIDS_S8['S8_err_dn']
    chi2_kids = (dS8_kids / err_kids)**2

    # DES Y3
    chi2_des = ((S8_model - DES_S8['S8']) / DES_S8['S8_err'])**2

    return chi2_kids + chi2_des

def total_chi2(params):
    """Total chi-squared for SSR parameter vector."""
    Xi_interp, dXi_interp = solve_background(params)
    z_growth = np.array([0.0, 0.1, 0.2, 0.38, 0.51, 0.61, 0.71, 0.93, 1.0])
    growth = compute_growth(params, Xi_interp, z_arr=z_growth)

    c2_bao = chi2_BAO(params, Xi_interp, dXi_interp)
    c2_rsd = chi2_RSD(params, Xi_interp, growth)
    c2_s8  = chi2_S8(params, Xi_interp, growth)

    return c2_bao + c2_rsd + c2_s8, c2_bao, c2_rsd, c2_s8

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PARAMETER SCAN (proxy for MCMC marginalisation)
# ═══════════════════════════════════════════════════════════════════════════════

def parameter_scan():
    """
    Grid scan over (lambda, beta) to find best-fit region.
    This is the computationally feasible proxy for MCMC in this environment.
    Full MCMC with Cobaya is set up in Section 8.
    """
    lambda_arr = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    beta_arr   = np.array([0.05, 0.10, 0.15, 0.20, 0.30])

    results = []
    print(f"{'lambda':>8} {'beta':>6} {'chi2_tot':>10} {'chi2_BAO':>10} {'chi2_RSD':>10} {'chi2_S8':>10} {'S8_model':>10}")
    print("-" * 72)

    best_chi2 = 1e10
    best_params = None

    for lam in lambda_arr:
        for beta in beta_arr:
            params = {**SSR_FIDUCIAL, 'lambda': lam, 'beta': beta}
            c2_tot, c2_bao, c2_rsd, c2_s8 = total_chi2(params)

            Xi_interp, _ = solve_background(params)
            z_g = np.array([0.0, 0.38, 0.51, 0.61])
            gr  = compute_growth(params, Xi_interp, z_arr=z_g)
            S8  = compute_S8(params, Xi_interp, gr)

            results.append({
                'lambda': lam, 'beta': beta,
                'chi2': c2_tot, 'chi2_BAO': c2_bao,
                'chi2_RSD': c2_rsd, 'chi2_S8': c2_s8,
                'S8': S8
            })

            marker = " <-- BEST" if c2_tot < best_chi2 else ""
            if c2_tot < best_chi2:
                best_chi2 = c2_tot
                best_params = params.copy()

            print(f"{lam:>8.2f} {beta:>6.2f} {c2_tot:>10.2f} {c2_bao:>10.2f} {c2_rsd:>10.2f} {c2_s8:>10.2f} {S8:>10.4f}{marker}")

    return results, best_params

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: COBAYA MCMC INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

COBAYA_INFO = """
# Cobaya MCMC configuration for SSR full parameter constraints
# Run with: cobaya-run ssr_cobaya.yaml
# Requires: cobaya, GetDist, Planck 2018 likelihood, DESI DR1 likelihood

theory:
  ssr_boltzmann:
    external: ssr_cobaya_theory.py
    speed: 2

likelihood:
  planck_2018_lowl.TT:
  planck_2018_lowl.EE:
  planck_2018_highl_plik.TTTEEE:
  bao.desi_dr1_bgs:
  bao.desi_dr1_lrg1:
  bao.desi_dr1_lrg2:
  bao.desi_dr1_lrg3:
  bao.desi_dr1_elg:
  bao.desi_dr1_qso:

params:
  # LCDM baseline
  H0:
    prior: {min: 60, max: 80}
    latex: H_0
  Omega_m:
    prior: {min: 0.2, max: 0.5}
    latex: \\Omega_m
  sigma8:
    prior: {min: 0.6, max: 1.0}
    latex: \\sigma_8

  # SSR parameters
  ssr_lambda:
    prior: {min: 0.01, max: 5.0, dist: log-normal, loc: -1, scale: 1}
    latex: \\lambda
  ssr_beta:
    prior: {min: 0.001, max: 1.0, dist: log-normal, loc: -2, scale: 1}
    latex: \\beta
  ssr_Lambda4:
    prior: {min: 1e-6, max: 1e-2, dist: log-normal, loc: -4, scale: 1}
    latex: \\Lambda^4
  ssr_f_Xi:
    prior: {min: 0.01, max: 1.0}
    latex: f_\\Xi
  ssr_A:
    prior: {min: -0.05, max: 0.05}
    latex: A
  ssr_omega:
    prior: {min: 0.1, max: 10.0}
    latex: \\omega
  ssr_phi:
    prior: {min: 0, max: 6.28}
    latex: \\phi

sampler:
  mcmc:
    Rminus1_stop: 0.01
    max_tries: 1000

output: chains/ssr_mcmc
"""

SSR_COBAYA_THEORY = '''
"""
SSR Cobaya theory class for MCMC parameter inference.
Drop this file alongside ssr_cobaya.yaml and run with cobaya-run.
"""
from cobaya.theory import Theory
import numpy as np
from ssr_module import solve_background, H_SSR, comoving_distance, compute_growth, compute_S8

class SSRBoltzmann(Theory):
    def initialize(self):
        self.z_arr = np.linspace(0, 3, 200)
        self.k_arr = np.logspace(-3, 1, 100)

    def get_requirements(self):
        return {}

    def calculate(self, state, want_derived=True, **params_values):
        params = {
            'lambda':   params_values['ssr_lambda'],
            'beta':     params_values['ssr_beta'],
            'Lambda4':  params_values['ssr_Lambda4'],
            'f_Xi':     params_values['ssr_f_Xi'],
            'n':        6.0,   # corrected: consistent with paper p_SSR = M_p^4/zeta^6
            'A':        params_values['ssr_A'],
            'omega':    params_values['ssr_omega'],
            'phi':      params_values['ssr_phi'],
        }
        Xi_interp, dXi_interp = solve_background(params)
        H_z = np.array([H_SSR(1/(1+z), params, Xi_interp, dXi_interp)
                        for z in self.z_arr])
        state['Hubble'] = {'z': self.z_arr, 'H': H_z}

        z_growth = np.linspace(0, 2, 50)
        growth = compute_growth(params, Xi_interp, z_arr=z_growth)
        state['growth'] = growth
        state['S8'] = compute_S8(params, Xi_interp, growth)

    def get_Hubble(self, z, units='km/s/Mpc'):
        return self._current_state['Hubble']

    def get_fsigma8(self, z):
        return np.interp(z, self._current_state['growth']['z'],
                         self._current_state['growth']['fsigma8'])
'''

if __name__ == "__main__":
    print("=" * 72)
    print("SSR NUMERICAL MODULE: PARAMETER SCAN AND CHI-SQUARED ANALYSIS")
    print("=" * 72)
    print()
    print("Computing background solutions and observational chi-squared...")
    print()

    results, best_params = parameter_scan()

    print()
    print("=" * 72)
    print("BEST-FIT PARAMETERS")
    print("=" * 72)
    for k, v in best_params.items():
        print(f"  {k:>12}: {v}")

    print()
    print("S8 TENSION ANALYSIS")
    print("=" * 72)
    print(f"  Planck 2018 S8 = {PLANCK_S8['S8']:.3f} +/- {PLANCK_S8['S8_err']:.3f}")
    print(f"  KiDS-1000   S8 = {KIDS_S8['S8']:.3f} +{KIDS_S8['S8_err_up']:.3f}/-{KIDS_S8['S8_err_dn']:.3f}")
    print(f"  DES Y3      S8 = {DES_S8['S8']:.3f} +/- {DES_S8['S8_err']:.3f}")
    print()

    Xi_best, dXi_best = solve_background(best_params)
    z_g = np.array([0.0, 0.1, 0.2, 0.38, 0.51, 0.61, 0.71, 0.93, 1.0])
    gr_best = compute_growth(best_params, Xi_best, z_arr=z_g)
    S8_best = compute_S8(best_params, Xi_best, gr_best)

    tension_planck = (PLANCK_S8['S8'] - S8_best) / PLANCK_S8['S8_err']
    print(f"  SSR best-fit S8 = {S8_best:.4f}")
    print(f"  Tension with Planck: {tension_planck:.2f} sigma")
    print(f"  S8 suppression vs LCDM: {(S8_best - sigma8_LCDM*np.sqrt(Omega_m0/0.3)):.4f}")

    print()
    print("P(k) SCALE-DEPENDENT FEATURE")
    print("=" * 72)
    Pk_best = compute_Pk(best_params, Xi_best,
                          k_arr_hMpc=np.logspace(-3, 1, 80), z=0.0)
    print(f"  k* (max suppression scale): {Pk_best['k_star']:.4f} h/Mpc")
    print(f"  Max P(k) suppression:       {Pk_best['max_suppression']*100:.2f}%")
    print(f"  Suppression at k=0.1 h/Mpc: {(1-float(np.interp(0.1, Pk_best['k'], Pk_best['Pk_ratio'])))*100:.2f}%")
    print(f"  Suppression at k=1.0 h/Mpc: {(1-float(np.interp(1.0, Pk_best['k'], Pk_best['Pk_ratio'])))*100:.2f}%")

    print()
    print("fsigma8 PREDICTIONS (best-fit SSR vs data)")
    print("=" * 72)
    print(f"{'z':>6} {'SSR fsigma8':>14} {'BOSS data':>12} {'DESI data':>12}")
    print("-" * 48)
    fs8_interp = np.interp
    for z_val in [0.38, 0.51, 0.61, 0.71, 0.93]:
        fs8_ssr = float(np.interp(z_val, gr_best['z'], gr_best['fsigma8']))
        boss_val = float(np.interp(z_val, BOSS_RSD['z'], BOSS_RSD['fsigma8'],
                         left=np.nan, right=np.nan)) if z_val <= 0.61 else np.nan
        desi_val = float(np.interp(z_val, DESI_RSD['z'], DESI_RSD['fsigma8'],
                         left=np.nan, right=np.nan))
        boss_str = f"{boss_val:.4f}" if not np.isnan(boss_val) else "  ---  "
        desi_str = f"{desi_val:.4f}" if not np.isnan(desi_val) else "  ---  "
        print(f"{z_val:>6.2f} {fs8_ssr:>14.4f} {boss_str:>12} {desi_str:>12}")

    print()
    print("COBAYA MCMC CONFIGURATION: saved to ssr_cobaya.yaml")
    print("SSR COBAYA THEORY CLASS:   saved to ssr_cobaya_theory.py")

    # Save Cobaya files
    with open("/home/claude/ssr_cobaya.yaml", "w") as f:
        f.write(COBAYA_INFO)
    with open("/home/claude/ssr_cobaya_theory.py", "w") as f:
        f.write(SSR_COBAYA_THEORY)

    print()
    print("=" * 72)
    print("DONE. All results ready for companion paper.")
    print("=" * 72)
