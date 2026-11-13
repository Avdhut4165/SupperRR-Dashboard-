"""Step 5: reliability (split-half, era split) and sensitivity of the available top 10."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import config as C
from scoring import score

def _raw_hp(d, mask):
    x = d[mask & d.PP]
    return x.groupby('pid').apply(lambda t: pd.Series({'n': len(t), 'm': np.average(t.v, weights=t.w) * 100}), include_groups=False)

def reliability(frames, n_rep=30, seed=C.SPLIT_SEED, min_n=40):
    rng = np.random.RandomState(seed); out = {}
    for k, d in frames.items():
        mids = d.match_id.unique(); cs, ss = [], []
        for _ in range(n_rep):
            h = pd.Series(rng.rand(len(mids)) < .5, index=mids); fl = d.match_id.map(h)
            j = _raw_hp(d, fl).join(_raw_hp(d, ~fl), lsuffix='A', rsuffix='B', how='inner')
            j = j[(j.nA >= min_n) & (j.nB >= min_n)]
            cs.append(np.corrcoef(j.mA, j.mB)[0, 1]); ss.append(spearmanr(j.mA, j.mB)[0])
        r = float(np.mean(cs))
        fl = d.year <= 2024
        j = _raw_hp(d, fl).join(_raw_hp(d, ~fl), lsuffix='A', rsuffix='B', how='inner'); j = j[(j.nA >= min_n) & (j.nB >= min_n)]
        out[k] = dict(split_half_pearson=r, split_half_spearman=float(np.mean(ss)), spearman_brown=2 * r / (1 + r),
                      era_2022_24_vs_2025_26_pearson=float(np.corrcoef(j.mA, j.mB)[0, 1]), era_n=len(j))
    return pd.DataFrame(out).T

def sensitivity(frames, b, avail, excluded):
    q = lambda p: b.LI.quantile(p)
    variants = {'Top 10% LI': dict(col='PP10'), 'Top 20% LI': dict(col='PP20'), 'Top 30% LI': dict(col='PP30'),
                'Chases only': dict(col='PPch'), 'Overs 7+ only': dict(col='PP7'), 'Wicket value x2': dict(wv2=True),
                'No opposition down-weight': dict(w1=True), 'IPL only': dict(scope='IPL', minb=60, minapp=10),
                'T20I only': dict(scope='T20I', minb=60, minapp=10), 'Weights 40/40/20': dict(wts=dict(L=.4, C=.4, D=.2)),
                'Level only': dict(wts=dict(L=1, C=0, D=0))}
    res = {}
    for k, d in frames.items():
        d = d.copy()
        d['PP10'] = d.LI >= q(.90); d['PP20'] = d.LI >= q(.80); d['PP30'] = d.LI >= q(.70)
        d['PPch'] = d.PP & (d.innings == 2); d['PP7'] = d.PP & (d.overid >= 7)
        if k == 'BAT': d['v_wv2'] = (d.bat_runs - d.E_bat_runs) - 2 * d.WV * (d.bat_out.astype(float) - d.E_bat_out)
        else: d['v_wv2'] = -(d.bowl_runs - d.E_bowl_runs) + 2 * d.WV * (d.bowl_wkt.astype(float) - d.E_bowl_wkt)
        a0 = avail[k]; top0 = set(a0.index[:10]); r0 = pd.Series(np.arange(len(a0)), index=a0.index); out = []
        for name, o in variants.items():
            dd = d.copy()
            if o.get('wv2'): dd['v'] = dd.v_wv2
            if o.get('w1'): dd['w'] = 1.0
            if 'scope' in o: dd = dd[dd.scope == o['scope']]
            e = score(dd, weights=o.get('wts'), pp_col=o.get('col', 'PP'), minb=o.get('minb', C.MIN_HP_BALLS), minapp=o.get('minapp', C.MIN_HP_APPS))
            e = e[~e.index.isin(excluded)]
            rr = pd.Series(np.arange(len(e)), index=e.index); com = rr.index.intersection(r0.index)
            out.append(dict(variant=name, overlap_with_base_top10=len(top0 & set(e.index[:10])),
                            spearman=spearmanr(r0[com], rr[com])[0] if len(com) > 5 else np.nan,
                            n_eligible_available=len(e), top3=', '.join(e.name.iloc[:3])))
        res[k] = pd.DataFrame(out)
    return res
