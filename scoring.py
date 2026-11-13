"""Step 3: player-level aggregation, empirical-Bayes shrinkage and the Pressure Reliability Score (PRS)."""
import numpy as np, pandas as pd
import config as C

def prepare(b):
    """Add pressure flag, weights and per-ball values; return (batting frame, bowling frame, threshold)."""
    b = b.copy()
    q = b.LI.quantile(C.HP_QUANTILE)
    b['w'] = np.where(b.tier_ok, 1.0, C.ASSOC_WEIGHT)
    b['PP'] = b.LI >= q
    b['v_bat'] = (b.bat_runs - b.E_bat_runs) - b.WV * (b.bat_out.astype(float) - b.E_bat_out)
    b['v_bowl'] = -(b.bowl_runs - b.E_bowl_runs) + b.WV * (b.bowl_wkt.astype(float) - b.E_bowl_wkt)
    bat = b[b.ext != 'wide'].copy(); bat['pid'] = bat.striker_id; bat['name'] = bat.striker_name; bat['v'] = bat.v_bat
    bow = b.copy(); bow['pid'] = bow.bowler_id; bow['name'] = bow.bowler_name; bow['v'] = bow.v_bowl
    return b, bat, bow, q

def _eb(mean, n, sig2):
    tau2 = max(np.var(mean, ddof=1) - sig2 * np.mean(1 / n), 1e-6)
    k = sig2 / tau2
    return mean * n / (n + k), k

def score(d, weights=None, mult=None, pp_col='PP', minb=C.MIN_HP_BALLS, minapp=C.MIN_HP_APPS):
    """Return a DataFrame of eligible players ranked by PRS. `mult` = match_id -> bootstrap multiplicity."""
    wts = weights or C.WEIGHTS
    d = d[['pid', 'name', 'match_id', 'innings', 'v', 'w', pp_col]].rename(columns={pp_col: 'hp'})
    if mult is not None:
        d = d.assign(m=d.match_id.map(mult).fillna(0)); d = d[d.m > 0]
    else:
        d = d.assign(m=1.0)
    d['ww'] = d.w * d.m; d['wv'] = d.ww * d.v
    g = d.groupby(['pid', 'hp']).agg(sw=('ww', 'sum'), swv=('wv', 'sum'), cnt=('m', 'sum')).unstack('hp').fillna(0)
    hpn, hpc, npn = g[('sw', True)], g[('cnt', True)], g[('sw', False)]
    hpm = g[('swv', True)] / hpn.replace(0, np.nan); npm = g[('swv', False)] / npn.replace(0, np.nan)
    a = d[d.hp].groupby(['pid', 'match_id', 'innings']).agg(n=('m', 'sum'), s=('wv', 'sum'), m=('m', 'first'))
    a = a[a.n >= C.MIN_APP_BALLS * a.m]; a['pos'] = (a.s > 0) * a.m
    ap = a.groupby('pid').agg(napp=('m', 'sum'), pos=('pos', 'sum'))
    ids = hpc[hpc >= minb].index.intersection(ap.index[ap.napp >= minapp])
    e = pd.DataFrame({'hp_n': hpn[ids], 'hp_balls': hpc[ids], 'hp_raw': hpm[ids] * 100, 'np_raw': npm[ids] * 100, 'np_n': npn[ids]}).join(ap)
    sig_hp = np.average((d[d.hp].v - d[d.hp].v.mean()) ** 2)
    sig_np = np.average((d[~d.hp].v - d[~d.hp].v.mean()) ** 2)
    e['hp_shr'], k1 = _eb(hpm[ids].values, hpn[ids].values, sig_hp)
    e['np_shr'], k2 = _eb(npm[ids].fillna(0).values, npn[ids].values, sig_np)
    e['hp_shr'] *= 100; e['np_shr'] *= 100
    p0 = e.pos.sum() / e.napp.sum()
    e['cons'] = (e.pos + C.CONS_PRIOR * p0) / (e.napp + C.CONS_PRIOR)
    e['delta'] = e.hp_shr - e.np_shr
    z = lambda s: (s - s.mean()) / s.std()
    e['zL'], e['zC'], e['zD'] = z(e.hp_shr), z(e.cons), z(e.delta)
    e['PRS'] = wts['L'] * e.zL + wts['C'] * e.zC + wts['D'] * e.zD
    e['name'] = d.groupby('pid').name.agg(lambda s: s.iloc[-1])[e.index]
    e = e.sort_values('PRS', ascending=False); e['rank'] = np.arange(1, len(e) + 1)
    e.attrs.update(k_hp=k1, k_np=k2, p0=p0)
    return e
