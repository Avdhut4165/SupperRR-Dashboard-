"""Locate the six input CSVs by their column signature (file names are hashed)."""
from pathlib import Path
import pandas as pd

def _cols(path):
    return set(pd.read_csv(path, nrows=0).columns)

def load_inputs(data_dir):
    data_dir = Path(data_dir)
    found = {}
    for p in sorted(data_dir.glob('*.csv')):
        c = _cols(p)
        if {'striker_id', 'innings_ball_num', 'bowler_id'} <= c: found['balls'] = p
        elif {'win_team_id', 'revised_target', 'venue_id'} <= c: found['matches'] = p
        elif {'comp_name', 'player_id', 'player_name'} <= c: found['squads'] = p
        elif {'is_impact', 'is_played', 'player_id'} <= c: found['xi'] = p
        elif {'comp_name', 'match_format'} <= c: found['comps'] = p
        elif {'team_name', 'team_id', 'comp_id'} <= c and 'player_id' not in c: found['teams'] = p
    need = {'balls', 'matches', 'comps', 'squads', 'teams'}
    missing = need - set(found)
    if missing:
        raise FileNotFoundError(f'Could not identify input file(s) {sorted(missing)} in {data_dir}. Found: {list(found)}')
    return {k: pd.read_csv(v) for k, v in found.items()}
