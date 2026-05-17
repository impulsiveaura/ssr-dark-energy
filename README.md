# SSR Dark Energy

Code for: **Sovereign Scaling Resonance (SSR): an Oscillatory Dark Energy 
Extension of ΛCDM with Cosmological Constraints and Laboratory Falsifiability**

**Author:** Sharaf Samiur Rahman — Independent Researcher, London, UK
**ORCID:** 0009-0003-4630-398X
**License:** MIT (code) / CC BY 4.0 (data and chains — see Zenodo)

## Data and chains
Full MCMC chain files: https://doi.org/10.5281/zenodo.XXXXXXX
(Replace XXXXXXX with your Zenodo DOI after Step 15 below)

## Preprint
arXiv link: [to be added after arXiv submission]

## Requirements
- Python 3.11+
- Cobaya >= 3.4
- NumPy, SciPy, Matplotlib, mpi4py

## How to run
```bash
cd ~/ssr_run
bash run_ssr_chains.sh
```

## Files
| File | Purpose |
|------|---------|
| ssr_module_v4.py | Background ODE, growth, chi2 functions |
| ssr_growth_like_v2.py | Cobaya likelihood class |
| ssr_cobaya_theory.py | Cobaya theory class |
| ssr_cobaya_v3.yaml | SSR MCMC configuration |
| lcdm_v2.yaml | ΛCDM reference configuration |
| generate_figures.py | H(z) and w(z) figure generation |
| run_ssr_chains.sh | Master launcher script |
