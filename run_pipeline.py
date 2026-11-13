#!/usr/bin/env python3
"""End-to-end pipeline: raw CSVs -> pressure model -> player ratings -> available top 10 + robustness.

Usage:
    python run_pipeline.py --data-dir /path/to/csvs --out-dir ./results [--n-boot 300] [--skip-robustness] [--no-charts]
"""
import argparse, json, sys, time, pickle
from pathlib import Path
import pandas as pd
import config as C
from io_utils import load_inputs
from prep import build_balls
from model import fit_models, add_wicket_value
from scoring import prepare, score
from availability import build_player_names, build_exclusions, conditional_ids, bootstrap_ranks, build_tables
from robustness import reliability, sensitivity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', default='results')
    ap.add_argument('--n-boot', type=int, default=300); ap.add_argument('--skip-robustness', action='store_true')
    ap.add_argument('--no-charts', action='store_true')
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); step = lambda s: print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)

    step('1/6 loading and cleaning data'); inp = load_inputs(a.data_dir); b = build_balls(inp)
    print(f'  {b.match_id.nunique()} matches, {len(b):,} deliveries, {b.scope.value_counts().to_dict()}')
    step('2/6 win-probability, leverage and baseline models (5-fold, grouped by match)'); b, metrics = fit_models(b)
    step('3/6 wicket value'); b = add_wicket_value(b)
    step('4/6 player scores'); b, bat, bow, q = prepare(b)
    print(f'  high-pressure threshold: leverage >= {q * 100:.2f} percentage points ({b.PP.mean():.1%} of deliveries)')
    frames = {'BAT': bat, 'BOWL': bow}
    full = {k: score(d) for k, d in frames.items()}
    print(f'  eligible players: batters {len(full["BAT"])}, bowlers {len(full["BOWL"])}')
    for k, e in full.items(): e.round(3).to_csv(out / f'full_league_{"batters" if k == "BAT" else "bowlers"}.csv')

    step('5/6 availability filter and bootstrap stability')
    reason, status, unmatched = build_exclusions(b, inp['squads'])
    if unmatched: print('  WARNING: names not found in data (check spelling/aliases in config.py):', unmatched)
    excluded = set(reason); cond = conditional_ids(b)
    nm = build_player_names(b).drop_duplicates('pid').set_index('pid').name
    pd.DataFrame({'pid': list(reason), 'name': [nm.get(p, '') for p in reason], 'reason': list(reason.values())}).to_csv(out / 'exclusion_reasons.csv', index=False)
    boot = bootstrap_ranks(frames, excluded, a.n_boot)
    avail, rows = build_tables(frames, full, excluded, cond, boot, a.n_boot)
    for k, nm in [('BAT', 'batters'), ('BOWL', 'bowlers')]:
        rows[k].head(10).round(2).to_csv(out / f'top10_{nm}_available.csv')
        avail[k].round(3).to_csv(out / f'all_available_{nm}.csv')
        print(f'\n  TOP 10 AVAILABLE {nm.upper()}:'); print(rows[k].head(10)[['Player', 'Role', 'HP_balls', 'PRS', 'Top10_boot_pct']].round(2).to_string())
    json.dump(metrics, open(out / 'model_metrics.json', 'w'), indent=2)

    if not a.skip_robustness:
        step('6/6 reliability and sensitivity')
        rel = reliability(frames); rel.round(3).to_csv(out / 'reliability.csv'); print(rel.round(3).to_string())
        sens = sensitivity(frames, b, avail, excluded)
        for k, v in sens.items(): v.round(3).to_csv(out / f'sensitivity_{"batters" if k == "BAT" else "bowlers"}.csv', index=False)
    if not a.no_charts:
        from charts import make_charts
        make_charts(b, q, full, avail, rows, out)
    b.to_pickle(out / 'balls_scored.pkl')
    step('done. outputs in ' + str(out.resolve()))

if __name__ == '__main__':
    main()
