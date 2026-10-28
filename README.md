# CD-SAGE pre-acceptance public snapshot

This directory is a scoped public snapshot for the manuscript *Continuous-Delay SAGE Algorithm for Underwater Acoustic Channel Estimation*. It contains the core Python implementation, a deterministic demonstration, selected regression tests, one representative Monte Carlo driver, and selected result figures.

## Included material

- `sage/`: core signal, estimator, evaluation, and bound routines. The frozen entry point is `sage.cd_sage_default.fit_cd_sage`.
- `scripts/quick_demo.py`: deterministic three-path example using the parameter setting reported in the manuscript.
- `scripts/mc_revision.py`: representative paired-observation Monte Carlo and accounting driver.
- `tests/`: selected checks for path assignment, error accounting, and joint CRB calculations.
- `figures/`: selected manuscript figures covering the algorithm flow, interpolation bias and variance, aggregate RMSE, ablation, measured-CIR replay, and enhanced baselines.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/quick_demo.py
python -m unittest discover -s tests -v
```

All synthetic delays are measured in samples. The demonstration is defined by relative delays and complex gains; it does not assign a unique physical source-receiver geometry.

## Scope of this snapshot

This pre-acceptance snapshot deliberately omits raw Watermark inputs, full Monte Carlo caches, internal audit records, manuscript sources, and several large experiment outputs. The complete project package, including all simulation and baseline scripts, configurations, indexed CIR inputs, and result tables, will be released after the paper is accepted.

Watermark source CIRs remain subject to the original provider's terms and are not redistributed here. See `LICENSE_NOTICE.md` for the current code-release status.

## Figure index

- `Fig_01_flowchart.pdf`: CD-SAGE processing flow.
- `Fig_05_bias.pdf`: finite-lag interpolation bias.
- `Fig_06_variance.pdf`: variance, MSE, and local reference.
- `Fig_07_rmse.pdf`: aggregate delay, magnitude, and phase RMSE.
- `Fig_11_ablation.pdf`: estimator-component ablation.
- `Fig_12_watermark.pdf`: NOF1/NCS1 measured-CIR replay.
- `Fig_14_enhanced.pdf`: refined and off-grid baseline comparison.

## Citation

Please cite the accompanying manuscript if you use this snapshot. Final bibliographic details will be added after publication.
