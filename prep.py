"""Step 1: clean ball-by-ball data and derive match-state features."""
import numpy as np, pandas as pd
import config as C

def build_balls(inp):
    b, m, comps, teams = inp['balls'].copy(), inp['matches'], inp['comps'], inp['teams']
    for k in ['is_four', 'is_six', 'is_wicket']:
        b[k] = b[k].astype(str).str.lower().eq('true')
    m = m.merge(comps, on='comp_id')
    m['scope'] = np.where(m.comp_name.str.startswith('IPL'), 'IPL', np.where(m.match_format == 'T20I', 'T20I', 'OTHER'))
    # keep IPL + T20I; drop DLS-affected and no-result matches
    keep = m[m.scope.isin(['IPL', 'T20I']) & m.revised_target.isna() & m.first_inn_revised_overs.isna()
             & m.second_inn_revised_overs.isna() & m.win_team_id.notna()]
    b = b[b.match_id.isin(keep.match_id) & (b.innings <= 2)].copy()      # innings 3/4 = super overs
    b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

    b['ext'] = b.extras_type.fillna('none')
    xr = {'none': 0, 'wide': 1, 'legbyes': C.BYE_RUNS, 'byes': C.BYE_RUNS, 'Overstep': 1, 'other': 1, 'Penalty': 5}
    b['extra_runs'] = b.ext.map(xr).fillna(1)
    b['team_runs'] = b.runs + b.extra_runs
    b['legal'] = ~b.ext.isin(['wide', 'Overstep'])
    b['bat_runs'] = b.runs
    b['bowl_runs'] = b.runs + np.where(b.ext.isin(['wide', 'Overstep']), 1, 0)   # byes/leg-byes not charged to bowler
    b['team_wkt'] = b.is_wicket & (b.dismissal_type != 'Retire Out')
    b['bat_out'] = b.team_wkt & (b.dismissed_player_id == b.striker_id)
    b['bowl_wkt'] = b.team_wkt & b.dismissal_type.isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'])

    g = b.groupby(['match_id', 'innings'])
    b['runs_before'] = g.team_runs.cumsum() - b.team_runs
    b['wkts_before'] = g.team_wkt.cumsum() - b.team_wkt.astype(int)
    b['legal_before'] = g.legal.cumsum() - b.legal.astype(int)
    b['balls_left'] = (120 - b.legal_before).clip(lower=0)
    tot = g.team_runs.sum().unstack(); tot.columns = ['i1', 'i2']
    b = b.merge(tot['i1'].rename('target_base'), left_on='match_id', right_index=True)
    b['target'] = np.where(b.innings == 2, b.target_base + 1, np.nan)
    b = b.merge(keep[['match_id', 'scope', 'win_team_id', 'venue_id', 'knock_out', 'stage',
                      'home_team_id', 'away_team_id', 'match_date', 'comp_name']], on='match_id')
    b['bat_win'] = (b.batting_team_id == b.win_team_id).astype(int)
    b['phase'] = pd.cut(b.overid, [0, 6, 15, 20], labels=['PP', 'MID', 'DEATH'])

    names = teams.drop_duplicates('team_id').set_index('team_id').team_name
    hn, an = b.home_team_id.map(names), b.away_team_id.map(names)
    b['tier_ok'] = (b.scope == 'IPL') | (hn.isin(C.FULL_MEMBERS) & an.isin(C.FULL_MEMBERS))
    b['year'] = pd.to_datetime(b.match_date).dt.year
    return b
