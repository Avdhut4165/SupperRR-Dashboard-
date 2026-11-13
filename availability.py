"""Step 4: availability / eligibility filter, bootstrap stability and player tables."""
import numpy as np, pandas as pd
import config as C
from scoring import score

def build_player_names(b):
    n = pd.concat([b[['striker_id', 'striker_name']].set_axis(['pid', 'name'], axis=1),
                   b[['bowler_id', 'bowler_name']].set_axis(['pid', 'name'], axis=1)]).drop_duplicates()
    n['n'] = n.name.map(C.norm)
    return n

def _find(names, full_name):
    m = names[names.n == C.norm(full_name)]
    return m.pid.unique().tolist()

def build_exclusions(b, squads):
    """Return (reason dict pid->str, retained-status dict pid->team, list of unmatched names)."""
    names = build_player_names(b)
    reason, status, unmatched = {}, {}, []
    last = pd.concat([b.groupby('striker_id').match_date.max(), b.groupby('bowler_id').match_date.max()]).groupby(level=0).max()
    for pid in last[pd.to_datetime(last) < C.INACTIVE_BEFORE].index:
        reason[pid] = f'No IPL/T20I match since {C.INACTIVE_BEFORE[:4]} in data (retired or inactive)'
    nat = squads.groupby('player_id').team_name.agg(set)
    for nation, why in C.EXCLUDE_NATIONS.items():
        for pid, v in nat.items():
            if nation in v: reason[pid] = why
    for nm in C.RETIRED:
        ids = _find(names, nm)
        if not ids: unmatched.append(nm)
        for pid in ids: reason[pid] = 'Retired'
    for team, lst in C.RETAINED.items():
        for nm in lst:
            ids = _find(names, C.NAME_ALIASES.get(nm, nm))
            if not ids: unmatched.append(nm)
            for pid in ids:
                status[pid] = team
                reason[pid] = f'Retained/traded: {team}'
    return reason, status, unmatched

def conditional_ids(b):
    names = build_player_names(b)
    return [p for nm in C.CONDITIONAL for p in _find(names, nm)]

def bootstrap_ranks(frames, excluded, n_boot, seed=C.BOOT_SEED):
    """Resample matches with replacement; record each player's rank among AVAILABLE players."""
    rng = np.random.RandomState(seed)
    mids = np.unique(np.concatenate([f.match_id.unique() for f in frames.values()]))
    boot = {k: {} for k in frames}
    for i in range(n_boot):
        mult = pd.Series(rng.choice(mids, len(mids), replace=True)).value_counts()
        for k, d in frames.items():
            e = score(d, mult=mult); e = e[~e.index.isin(excluded)]
            for j, pid in enumerate(e.index): boot[k].setdefault(pid, []).append(j + 1)
    return boot

def _mode(s):
    s = s.dropna(); return s.mode().iloc[0] if len(s) else ''

def build_tables(frames, full, excluded, cond, boot, n_boot, topn=12):
    avail, rows = {}, {}
    for k, d in frames.items():
        a = full[k][~full[k].index.isin(excluded)].copy()
        a['avail_rank'] = np.arange(1, len(a) + 1); a['conditional'] = a.index.isin(cond)
        avail[k] = a
        out = []
        for pid, r in a.head(topn).iterrows():
            x = d[d.pid == pid]; hp = x[x.PP]; nn = x[~x.PP]; rk = np.array(boot[k].get(pid, [999]))
            row = dict(pid=pid, Player=r['name'])
            if k == 'BAT':
                hand = _mode(x.batting_type); order = x.batting_order.median()
                row.update(Role=f"{'LHB' if str(hand).startswith('L') else 'RHB'}, bats ~{int(round(order))}", HP_balls=len(hp),
                           HP_SR=hp.bat_runs.sum() / max(len(hp), 1) * 100, NP_SR=nn.bat_runs.sum() / max(len(nn), 1) * 100,
                           HP_balls_per_dismissal=len(hp) / max(hp.bat_out.sum(), 1))
            else:
                lh, ln = hp[hp.legal], nn[nn.legal]
                row.update(Role=_mode(x.bowling_type), HP_balls=len(hp), HP_econ=hp.bowl_runs.sum() / max(len(lh), 1) * 6,
                           NP_econ=nn.bowl_runs.sum() / max(len(ln), 1) * 6, HP_wkts=int(hp.bowl_wkt.sum()),
                           HP_dot_pct=(hp.bowl_runs == 0).mean() * 100)
            row.update(Matches=int(r.napp), Value_HP=r.hp_shr, Value_NP=r.np_shr, Consistency=r.cons * 100, Delta=r.delta,
                       PRS=r.PRS, Full_league_rank=int(r['rank']), Conditional=bool(r.conditional),
                       IPL_share=(hp.scope == 'IPL').mean() * 100, Top10_boot_pct=(rk <= 10).sum() / n_boot * 100,
                       Rank_p10_p90=f"{int(np.percentile(rk, 10))}-{int(np.percentile(rk, 90))}")
            out.append(row)
        rows[k] = pd.DataFrame(out); rows[k].index = range(1, len(out) + 1)
    return avail, rows
