# SupeRR Selector S2 — IPL Pressure Dashboard

This repository contains the pressure-rating pipeline, the supplied raw CSV inputs, verified reference outputs, the latest run outputs, and a static dashboard deployed through GitHub Pages.

## Open the dashboard

After the GitHub Pages workflow completes, the dashboard will be available at:

`https://avdhut4165.github.io/SupperRR-Dashboard-/`

The workflow publishes `site/index.html` after pushes to `main`.

## Run the pipeline

Python 3.10+ is required.

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
python run_pipeline.py --data-dir data/raw_data --out-dir results
```

Options include `--n-boot 300`, `--skip-robustness`, and `--no-charts`.

## Pipeline files

- `run_pipeline.py`: command line entry point
- `prep.py`, `model.py`, `scoring.py`, `availability.py`, `robustness.py`: pipeline stages
- `config.py`: modelling and availability settings
- `io_utils.py`: input discovery by CSV columns
- `charts.py`: optional output figures
- `data/raw_data/`: six input CSV files
- `reference_results/`: verified run outputs
- `results/`: latest run outputs, including the standalone dashboard and figures
- `site/index.html`: self-contained static dashboard for GitHub Pages

The latest run processed 788 matches and 188,151 deliveries. Win-probability log-loss was 0.5398 (baseline 0.6930). Reliability is modest; treat rankings as a shortlist.
