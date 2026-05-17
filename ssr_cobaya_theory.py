from cobaya.theory import Theory
import numpy as np, sys, importlib.util
sys.path.insert(0, ".")

# Load ssr_module_v4 explicitly so we never accidentally get the old v1
_spec = importlib.util.spec_from_file_location("ssr_module_v4", "ssr_module_v4.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
solve_background   = _m.solve_background
H_SSR              = _m.H_SSR
compute_growth     = _m.compute_growth
luminosity_distance = _m.luminosity_distance

_OMEGA_R0_H2 = 2.469e-5   # Omega_r0 * h^2

class SSRBoltzmann(Theory):

    params = {
        "H0": None, "Omega_m": None,
        "ssr_lambda": None, "ssr_beta": None, "ssr_Lambda4": None,
        "ssr_f_Xi": None, "ssr_A": None, "ssr_omega": None, "ssr_phi": None,
    }

    def initialize(self):
        self.z_arr = np.linspace(0.01, 2.5, 300)

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        return {}

    def calculate(self, state, want_derived=True, **params_values):
        H0      = params_values.get("H0", 67.4)
        Omega_m = params_values.get("Omega_m", 0.315)
        h       = H0 / 100.0
        Omega_r = _OMEGA_R0_H2 / h**2
        Omega_L = 1.0 - Omega_m - Omega_r

        p = {
            "H0": H0, "Omega_m": Omega_m,
            "Omega_r": Omega_r, "Omega_L": Omega_L,
            "lambda":  params_values["ssr_lambda"],
            "beta":    params_values["ssr_beta"],
            "Lambda4": params_values["ssr_Lambda4"],
            "f_Xi":    params_values["ssr_f_Xi"],
            "n":       6.0,
            "A":       params_values["ssr_A"],
            "omega":   params_values["ssr_omega"],
            "phi":     params_values["ssr_phi"],
        }
        try:
            Xi, dXi = solve_background(p)
            H_z  = np.array([H_SSR(1/(1+z), p, Xi, dXi) for z in self.z_arr])
            DL_z = np.array([luminosity_distance(z, p, Xi, dXi) for z in self.z_arr])
            DA_z = DL_z / (1.0 + self.z_arr)**2
            state["Hubble"]                    = {"z": self.z_arr, "H": H_z}
            state["luminosity_distance"]       = {"z": self.z_arr, "lum_dist": DL_z}
            state["angular_diameter_distance"] = {"z": self.z_arr, "ang_diam_dist": DA_z}
            state["_p"]   = p
            state["_Xi"]  = Xi
            state["_dXi"] = dXi
        except Exception as e:
            return False

    def get_Hubble(self, z, units="km/s/Mpc"):
        return np.interp(np.atleast_1d(z),
                         self._current_state["Hubble"]["z"],
                         self._current_state["Hubble"]["H"])

    def get_luminosity_distance(self, z):
        s = self._current_state["luminosity_distance"]
        return np.interp(np.atleast_1d(z), s["z"], s["lum_dist"])

    def get_angular_diameter_distance(self, z):
        s = self._current_state["angular_diameter_distance"]
        return np.interp(np.atleast_1d(z), s["z"], s["ang_diam_dist"])
