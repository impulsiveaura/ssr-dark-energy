from cobaya.likelihood import Likelihood
import numpy as np
import sys
sys.path.insert(0, '/Users/sharafsrahman/ssr_run')
import ssr_module
from ssr_module import solve_background, compute_growth, chi2_BAO, chi2_S8
from scipy.interpolate import CubicSpline

DESI_RSD_ONLY = {
    'z':    np.array([0.510, 0.706, 0.934]),
    'fs8':  np.array([0.455, 0.420, 0.408]),
    'err':  np.array([0.032, 0.029, 0.031]),
}

def chi2_RSD_desi(params, Xi_interp, growth_results):
    z_gr = growth_results['z']
    fs8_gr = growth_results['fsigma8']
    sort_idx = np.argsort(z_gr)
    fs8_interp = CubicSpline(z_gr[sort_idx], fs8_gr[sort_idx])
    chi2 = 0.0
    for i, z in enumerate(DESI_RSD_ONLY['z']):
        if z <= z_gr.max():
            chi2 += ((float(fs8_interp(z)) - DESI_RSD_ONLY['fs8'][i]) / DESI_RSD_ONLY['err'][i])**2
    return chi2

class SSRGrowthLikeDESI(Likelihood):
    params = {'H0': None, 'Omega_m': None, 'ssr_lambda': None, 'ssr_beta': None,
              'ssr_Lambda4': None, 'ssr_f_Xi': None, 'ssr_A': None, 'ssr_omega': None, 'ssr_phi': None}
    def get_requirements(self): return {}
    def logp(self, **p):
        ssr_module.H0_fid = p.get('H0', 67.4)
        ssr_module.Omega_m0 = p.get('Omega_m', 0.315)
        ssr_module.Omega_L0 = 1.0 - ssr_module.Omega_m0 - ssr_module.Omega_r0
        params = {'lambda': p['ssr_lambda'], 'beta': p['ssr_beta'],
                  'Lambda4': p['ssr_Lambda4'], 'f_Xi': p['ssr_f_Xi'],
                  'n': 4.6, 'A': p['ssr_A'], 'omega': p['ssr_omega'], 'phi': p['ssr_phi']}
        try:
            Xi, dXi = solve_background(params)
            z_g = np.array([0.0, 0.1, 0.2, 0.51, 0.71, 0.93, 1.0])
            growth = compute_growth(params, Xi, z_arr=z_g)
            return -0.5 * (chi2_BAO(params, Xi, dXi) + chi2_RSD_desi(params, Xi, growth) + chi2_S8(params, Xi, growth))
        except: return -1e30

class LCDMGrowthLikeDESI(Likelihood):
    params = {'H0': None, 'Omega_m': None}
    def get_requirements(self): return {}
    def logp(self, **p):
        ssr_module.H0_fid = p.get('H0', 67.4)
        ssr_module.Omega_m0 = p.get('Omega_m', 0.315)
        ssr_module.Omega_L0 = 1.0 - ssr_module.Omega_m0 - ssr_module.Omega_r0
        params = {'lambda': 0.0, 'beta': 0.0, 'Lambda4': 9.6e-5, 'f_Xi': 0.5,
                  'n': 4.6, 'A': 0.0, 'omega': 5.0, 'phi': 0.0}
        try:
            Xi, dXi = solve_background(params)
            z_g = np.array([0.0, 0.1, 0.2, 0.51, 0.71, 0.93, 1.0])
            growth = compute_growth(params, Xi, z_arr=z_g)
            return -0.5 * (chi2_BAO(params, Xi, dXi) + chi2_RSD_desi(params, Xi, growth) + chi2_S8(params, Xi, growth))
        except: return -1e30
