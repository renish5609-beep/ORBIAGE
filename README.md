# ORBIAGE — Improving Satellite Anomaly Detection by Accounting for AI Uncertainty

Reproducible research prototype for the Stardust Challenge.

## What it tests

Does uncertainty- and distribution-shift-aware ranking help prioritize anomalies in spacecraft telemetry when human review capacity is limited?

This experiment uses **ESA OPS-SAT-AD engineered segment features**, not raw time series. The reference dataset has 2,123 labeled segments, split into 1,594 training and 529 test segments.

**Important finding:** A conventional random forest outperformed the preliminary ORBIAGE score blends on the held-out set. The tuned uncertainty blend selected weight zero, so it is effectively the calibrated baseline. Selective abstention and human-review routing are proposed future work, not implemented capabilities.

## Setup

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt -e .
python -m pytest -q
python run_experiment.py --data data/dataset.csv --output results
```

Download the original `dataset.csv` from the [OPS-SAT-AD dataset](https://zenodo.org/records/12588359) and place it in `data/`. The dataset is intentionally **not** committed until its redistribution permissions can be confirmed.

## Repository structure

- `run_experiment.py`: reproducible model training, held-out evaluation and alert-budget figure.
- `src/orbiage/metrics.py`: testable ranking metrics.
- `tests/`: five unit tests.
- `.github/workflows/tests.yml`: automated CI tests.
- `results/`: reported summary table; regenerate from original dataset.
- `figures/` and `paper/`: supplementary binary artifacts available in the submission package.

## Representative held-out findings

| Method | ROC-AUC | Average precision | TP in top 100 |
|---|---:|---:|---:|
| Random forest | 0.9877 | 0.9676 | 97 |
| Sigmoid-calibrated RF | 0.9858 | 0.9652 | 97 |
| ORBIAGE shift blend | 0.9866 | 0.9664 | 95 |

These are preliminary single-split results, not independent validation or spacecraft fault-severity claims.

## Research provenance

Based on the OPS-SAT-AD benchmark described by Ruszczak et al., *Scientific Data* (2025), [DOI:10.1038/s41597-025-05035-3](https://doi.org/10.1038/s41597-025-05035-3). This independent student project is not affiliated with ESA.
