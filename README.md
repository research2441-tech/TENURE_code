# TENURE

Handshake-aware co-optimisation of cluster-head tenure, rekeying and trust-weighted routing for post-quantum secure wireless sensor networks.

This folder contains the complete simulator, experiment driver, statistics and figure scripts used in the paper. All files sit in one flat folder.

## Files

| File | Purpose |
|---|---|
| `tenure_sim.py` | Round-based IEEE 802.15.4 WSN simulator: O-QPSK link model, 6LoWPAN fragmentation with per-fragment ARQ, nRF52840 energy constants, Cortex-M4 cycle counts (pqm4, X25519, RSA-2048, Ascon), TENURE and five baselines (FixedTenure-PQ, DEEC-PQ, PQ-LEACH, pqDTLS-LEACH, MH-RSA), attack models and the mixture e-process. |
| `run_experiments.py` | Experiment driver for tuning, E1 (main), E2 (ablation), E3 (tenure sensitivity), E4 (robustness), E5 (generalisation), E10 (legacy migration). |
| `detector_bench.py` | E8: anytime-valid e-process versus CUSUM and fixed-rate rules (false alarms over horizon, detection delay). |
| `analyze.py` | Statistics (bootstrap CIs, Friedman, Holm-corrected Wilcoxon, Cliff's delta), figures 3-8 and `analysis.json`. |
| `fig_diagrams.py` | Figures 1-2 (architecture, key-establishment sequence). |
| `TENURE_Colab.ipynb` | Self-contained notebook for Google Colab (free tier). |
| `*.json` | Raw per-run results of the reported experiments (`E1_main.json` ... `E10_legacy.json`, `E8_detector.json`, `tune_fixedT.json`, `tuned.json`) and the analysis summary `analysis.json`. |
| `fig*.png` | Figures as used in the manuscript. |

## Reproduce

```bash
pip install -r requirements.txt
python run_experiments.py          # full protocol; QUICK=1 for a short run
python detector_bench.py
python analyze.py
python fig_diagrams.py
```

Environment variables: `TENURE_OUT` (results folder, default: current folder), `TENURE_FIG` (figure folder, default: current folder), `QUICK=1` (reduced seeds), `NPROC` (worker processes).
Existing result files are reused, so running `python analyze.py` in this folder regenerates every figure and `analysis.json` from the shipped results; delete a result file to recompute that experiment.

Seeds: test seeds 1-30, tuning seeds 101-110 (used only to select the fixed period of FixedTenure-PQ).

## Sources of constants

* ML-KEM / ML-DSA cycles: pqm4 benchmarks (Cortex-M4, m4fstack / m4f).
* X25519 cycles: Lenngren, IACR ePrint 2025/523.
* RSA-2048: CycloneCRYPTO 2.5.0 benchmark on a Cortex-M4 at 120 MHz (Oryx Embedded).
* Ascon: ascon-c reference implementation, 32-bit ARMv7 rows.
* Radio and CPU currents: nRF52840 product specification.

## License

MIT
