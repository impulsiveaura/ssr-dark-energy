#!/usr/bin/env python3
"""
SSR vs ΛCDM nested sampling — run overnight.
Place this file in the same directory as:
  ssr_module.py
  ssr_module_v4.py  (or whichever version you have)
  ssr_growth_like_desi.py

Then run:
  pip install ultranest cobaya camb --break-system-packages
  python3 run_nested_overnight.py | tee nested_log.txt
"""

import sys, os, numpy as np
from pathlib import Path

# ── Adjust this to wherever your SSR code lives ──────────────────────────────
SSR_DIR = str(Path(__file__).parent)
sys.path.insert(0, SSR_DIR)
RESULTS_DIR = os.path.join(SSR_DIR, 'nested_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from ssr_growth_like_desi import SSRGrowthLikeDESI, LCDMGrowthLikeDESI
import ultranest

print("Initialising likelihoods...")
ssr_like  = SSRGrowthLikeDESI();  ssr_like.initialize()
lcdm_like = LCDMGrowthLikeDESI(); lcdm_like.initialize()
print("Done.\n")

# ── SSR: 8 free parameters ────────────────────────────────────────────────────
param_names = ['H0','Omega_m','ssr_lambda','ssr_beta',
               'ssr_f_Xi','ssr_A','ssr_omega','ssr_phi']
mins = [60,  0.20, 0.0, 0.0, 0.01, -0.1, 0.5, 0.0  ]
maxs = [80,  0.50, 5.0, 2.0, 1.0,   0.1, 8.0, 6.283]

def prior_ssr(cube):
    return np.array([mn + c*(mx-mn) for c,mn,mx in zip(cube, mins, maxs)])

def loglike_ssr(params):
    d = dict(zip(param_names, params))
    d['ssr_Lambda4'] = 9.6e-5
    lp = ssr_like.logp(**d)
    return float(lp) if np.isfinite(lp) else -1e30

# ── ΛCDM: 2 free parameters ──────────────────────────────────────────────────
def prior_lcdm(cube):
    return np.array([60 + cube[0]*20, 0.20 + cube[1]*0.30])

def loglike_lcdm(params):
    lp = lcdm_like.logp(H0=params[0], Omega_m=params[1])
    return float(lp) if np.isfinite(lp) else -1e30

# ── Run SSR ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Run 1/2: SSR nested sampling (8 parameters)")
print("Estimated time: 3-8 hours depending on machine speed")
print("=" * 60)

sampler_ssr = ultranest.ReactiveNestedSampler(
    param_names, loglike_ssr, prior_ssr,
    log_dir=os.path.join(RESULTS_DIR, 'ssr'),
    resume='overwrite'
)
result_ssr = sampler_ssr.run(
    min_num_live_points=200,        # robust convergence
    max_num_improvement_loops=5,
    show_status=True,
    viz_callback=False
)
logZ_ssr     = result_ssr['logz']
logZ_ssr_err = result_ssr['logzerr']
print(f"\nSSR: log Z = {logZ_ssr:.3f} ± {logZ_ssr_err:.3f}  (ncall={result_ssr['ncall']})")

# ── Run ΛCDM ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Run 2/2: ΛCDM nested sampling (2 parameters)")
print("Estimated time: ~5 minutes")
print("=" * 60)

sampler_lcdm = ultranest.ReactiveNestedSampler(
    ['H0','Omega_m'], loglike_lcdm, prior_lcdm,
    log_dir=os.path.join(RESULTS_DIR, 'lcdm'),
    resume='overwrite'
)
result_lcdm = sampler_lcdm.run(
    min_num_live_points=200,
    max_num_improvement_loops=5,
    show_status=True,
    viz_callback=False
)
logZ_lcdm     = result_lcdm['logz']
logZ_lcdm_err = result_lcdm['logzerr']
print(f"\nΛCDM: log Z = {logZ_lcdm:.3f} ± {logZ_lcdm_err:.3f}  (ncall={result_lcdm['ncall']})")

# ── Bayes factor ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYES FACTOR RESULT")
print("=" * 60)
dlogZ = logZ_ssr - logZ_lcdm
err   = (logZ_ssr_err**2 + logZ_lcdm_err**2)**0.5
print(f"log B(SSR/ΛCDM) = {dlogZ:.2f} ± {err:.2f}")
print(f"B(SSR/ΛCDM)     = {np.exp(dlogZ):.4f}")
if   dlogZ >  5:  verdict = "Strong evidence FOR SSR"
elif dlogZ >  2.5: verdict = "Moderate evidence for SSR"
elif dlogZ > -2.5: verdict = "Inconclusive"
elif dlogZ > -5:  verdict = "Moderate evidence for ΛCDM"
else:              verdict = "Strong evidence FOR ΛCDM"
print(f"Verdict:          {verdict}")

np.save(os.path.join(RESULTS_DIR,'logZ_ssr.npy'),  [logZ_ssr,  logZ_ssr_err])
np.save(os.path.join(RESULTS_DIR,'logZ_lcdm.npy'), [logZ_lcdm, logZ_lcdm_err])
print(f"\nResults saved to {RESULTS_DIR}/")
print("Add log B(SSR/ΛCDM) to Section 3.3 of the paper before final submission.")
