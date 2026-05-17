from cobaya.likelihood import Likelihood
import numpy as np, sys, logging
sys.path.insert(0, ".")
from ssr_module_v4 import solve_background, compute_growth, chi2_BAO, chi2_RSD, chi2_S8

log = logging.getLogger(__name__)
_OMEGA_R0_H2 = 2.469e-5

class SSRGrowthLike(Likelihood):
    params = {
        "H0": None, "Omega_m": None,
        "ssr_lambda": None, "ssr_beta": None, "ssr_Lambda4": None,
        "ssr_f_Xi": None, "ssr_A": None, "ssr_omega": None, "ssr_phi": None,
    }

    def get_requirements(self): return {}

    def logp(self, **pv):
        H0 = pv["H0"]; Om = pv["Omega_m"]
        h  = H0 / 100.0
        Or = _OMEGA_R0_H2 / h**2
        p  = {"H0": H0, "Omega_m": Om, "Omega_r": Or, "Omega_L": 1.0 - Om - Or,
              "lambda": pv["ssr_lambda"], "beta": pv["ssr_beta"],
              "Lambda4": pv["ssr_Lambda4"], "f_Xi": pv["ssr_f_Xi"],
              "n": 6.0,
              "A": pv["ssr_A"], "omega": pv["ssr_omega"], "phi": pv["ssr_phi"]}
        z_g = np.array([0.0, 0.1, 0.2, 0.38, 0.51, 0.61, 0.71, 0.93, 1.0])
        try:
            Xi, dXi = solve_background(p)
            gr = compute_growth(p, Xi, z_arr=z_g)
            c2 = chi2_BAO(p, Xi, dXi) + chi2_RSD(p, Xi, gr) + chi2_S8(p, Xi, gr)
            return -1e30 if not np.isfinite(c2) else -0.5 * c2
        except ValueError as e:
            log.debug("ValueError: %s", e); return -1e30
        except Exception as e:
            log.warning("%s: %s", type(e).__name__, e); return -1e30
