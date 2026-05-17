#!/usr/bin/env bash
# =============================================================================
# run_ssr_chains.sh  —  SSR Cobaya MCMC launcher
# Always run from ~/ssr_run:   bash run_ssr_chains.sh
# =============================================================================

set -e

SSR_DIR="/Users/sharafsrahman/ssr_run"
PKG_DIR="/Users/sharafsrahman/cobaya_packages"

export PYTHONPATH="${SSR_DIR}:${PYTHONPATH}"
cd "${SSR_DIR}"

echo "Checking files..."
for f in ssr_module_v4.py ssr_growth_like_v2.py ssr_cobaya_theory.py; do
    [[ -f "$f" ]] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }
done

echo "Checking Pantheon data..."
[[ -f "$PKG_DIR/data/sn_data/Pantheon/full_long.dataset" ]] \
    && echo "  OK: Pantheon" \
    || { echo "  ERROR: Pantheon not found"; exit 1; }

echo "Writing yaml..."
python3 - << 'PYEOF'
import sys, yaml
sys.path.insert(0, '/Users/sharafsrahman/ssr_run')
info = {
    "theory": {"ssr_cobaya_theory.SSRBoltzmann": {"speed": 2}},
    "likelihood": {
        "sn.pantheon": {"path": "/Users/sharafsrahman/cobaya_packages/data/sn_data"},
        "ssr_growth_like_v2.SSRGrowthLike": None
    },
    "packages_path": "/Users/sharafsrahman/cobaya_packages",
    "params": {
        "H0":          {"prior": {"min": 60, "max": 80},    "ref": {"dist": "norm", "loc": 67.4,  "scale": 0.5},   "proposal": 0.4,   "latex": "H_0"},
        "Omega_m":     {"prior": {"min": 0.2, "max": 0.5},  "ref": {"dist": "norm", "loc": 0.315, "scale": 0.005}, "proposal": 0.004, "latex": "\\Omega_m"},
        "ssr_lambda":  {"prior": {"min": 0.01, "max": 5.0}, "ref": {"dist": "norm", "loc": 0.5,   "scale": 0.05},  "proposal": 0.04,  "latex": "\\lambda"},
        "ssr_beta":    {"prior": {"min": 0.001,"max": 1.0}, "ref": {"dist": "norm", "loc": 0.15,  "scale": 0.01},  "proposal": 0.008, "latex": "\\beta"},
        "ssr_Lambda4": {"prior": {"min": 1e-6, "max": 0.01},"ref": {"dist": "norm", "loc": 1e-4,  "scale": 5e-6},  "proposal": 4e-6,  "latex": "\\Lambda^4"},
        "ssr_f_Xi":    {"prior": {"min": 0.01, "max": 1.0}, "ref": {"dist": "norm", "loc": 0.5,   "scale": 0.05},  "proposal": 0.04,  "latex": "f_{\\Xi}"},
        "ssr_A":       {"prior": {"min": -0.1, "max": 0.1}, "ref": {"dist": "norm", "loc": 0.03,  "scale": 0.003}, "proposal": 0.002, "latex": "A"},
        "ssr_omega":   {"prior": {"min": 0.5,  "max": 8.0}, "ref": {"dist": "norm", "loc": 3.3,   "scale": 0.3},   "proposal": 0.2,   "latex": "\\omega"},
        "ssr_phi":     {"prior": {"min": 0.0,  "max": 6.2832},"ref":{"dist":"norm","loc":2.1,    "scale": 0.2},   "proposal": 0.15,  "latex": "\\phi"},
    },
    "sampler": {"mcmc": {"burn_in": 200, "max_tries": 10000,
        "Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.02,
        "learn_proposal": True, "proposal_scale": 1.9}},
    "output": "chains/ssr_v4"
}
with open('ssr_cobaya_v3.yaml', 'w') as f:
    yaml.dump(info, f, default_flow_style=False, allow_unicode=True)
print("  yaml OK")
PYEOF

echo "Sanity check..."
python3 - << 'PYEOF'
import sys, yaml
sys.path.insert(0, '/Users/sharafsrahman/ssr_run')
info = yaml.safe_load(open('ssr_cobaya_v3.yaml'))
from cobaya.model import get_model
model = get_model(info)
ref = dict(H0=67.4, Omega_m=0.315, ssr_lambda=0.5, ssr_beta=0.15,
           ssr_Lambda4=1e-4, ssr_f_Xi=0.5, ssr_A=0.03, ssr_omega=3.3, ssr_phi=2.1)
lp = model.logpost(ref)
print(f"  logpost = {lp:.4f}")
if lp <= -1e28:
    print("  ERROR: logpost is -1e30"); import sys; sys.exit(1)
print("  READY")
PYEOF

echo ""
echo "Launching chains (4 parallel)..."
mpirun -n 4 cobaya-run ssr_cobaya_v3.yaml --force 2>&1 | tee chains/ssr_v4.log
