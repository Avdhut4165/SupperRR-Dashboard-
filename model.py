"""Step 2: win-probability model, leverage (Pressure Index), expected-value baselines, wicket value.
All models are fitted out-of-fold, grouped by match, so no ball is scored by a model that saw its match."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss
import config as C

WP_PARAMS = dict(max_iter=80, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=600, l2_regularization=5)
GB_PARAMS = dict(max_iter=120, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=300, l2_regularization=3.0)
RUN_VALUES = np.array([0, 1, 2, 3, 4, 6])

def feats(inn, bl, w, r, tgt, sc, ven, over=None, bio=None, pace=None):
    inn = np.asarray(inn); bl = np.asarray(bl, float); w = np.asarray(w, float)
    r = np.asarray(r, float); tgt = np.asarray(tgt, float)
    played = np.maximum(120 - bl, 1)
    need = np.where(inn == 2, tgt - r, np.nan)
    rrr = np.where(inn == 2, need / np.maximum(bl, 1) * 6, np.nan)
    crr = (r + 8) / (played + 6) * 6          # smoothed: avoids spurious huge run rates on ball 1
    X = {'inn': inn, 'bl': bl, 'w': w, 'r': r, 'crr': crr, 'need': need, 'rrr': rrr,
         'rrr_minus_crr': rrr - crr, 'sc': sc, 'ven': ven}
    if over is not None:
        X.update({'over': over, 'bio': bio, 'pace': pace})
    return pd.DataFrame(X)

def wp_fun(mdl, inn, bl, w, r, tgt, sc, ven):
    """Win probability of the batting side, with terminal states handled exactly."""
    p = mdl.predict_proba(feats(inn, bl, w, r, tgt, sc, ven))[:, 1]
    inn = np.asarray(inn); r = np.asarray(r, float); tgt = np.asarray(tgt, float)
    w = np.asarray(w); bl = np.asarray(bl); chase = inn == 2
    p = np.where(chase & (r >= tgt), 1.0, p)
    p = np.where(chase & (r < tgt) & ((w >= 10) | (bl <= 0)), np.where(r == tgt - 1, 0.5, 0.0), p)
    return p

def add_context(b):
    b = b.copy()
    b['scope_i'] = (b.scope == 'IPL').astype(int)
    pos = set(b.pace_or_spin.dropna().unique())
    pace_labels = {x for x in pos if str(x).lower().startswith('pace')}
    if not pace_labels:
        raise ValueError(f"Unexpected pace_or_spin values {pos}; expected something starting with 'Pace'")
    b['pace'] = b.pace_or_spin.isin(pace_labels).astype(int)
    # venue scoring level: leave-one-match-out mean of first-innings totals
    i1 = b[b.innings == 1].groupby('match_id').agg(t=('team_runs', 'sum'), v=('venue_id', 'first'))
    vs = i1.groupby('v').t.agg(['sum', 'count']); i1 = i1.join(vs, on='v')
    i1['venue_avg'] = np.where(i1['count'] > 1, (i1['sum'] - i1.t) / (i1['count'] - 1).clip(lower=1), i1.t.mean())
    b = b.merge(i1['venue_avg'], left_on='match_id', right_index=True, how='left')
    b['venue_avg'] = b.venue_avg.fillna(i1.t.mean())
    tr = b.team_runs.clip(upper=6).replace({5: 4}).astype(int)
    b['oc'] = np.where(b.team_wkt, 6, tr.map({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}).fillna(0).astype(int))
    return b

def fit_models(b, verbose=True):
    b = add_context(b)
    Xs = feats(b.innings, b.balls_left, b.wkts_before, b.runs_before, b.target, b.scope_i, b.venue_avg)
    Xo = feats(b.innings, b.balls_left, b.wkts_before, b.runs_before, b.target, b.scope_i, b.venue_avg,
               b.overid, b.over_ball_num, b.pace)
    legal = b.legal.values; nonwide = (b.ext != 'wide').values
    n = len(b); wp = np.zeros(n); LI = np.zeros(n)
    E = {k: np.zeros(n) for k in ['bat_runs', 'bat_out', 'bowl_runs', 'bowl_wkt']}
    for f, (tr, te) in enumerate(GroupKFold(C.N_FOLDS).split(b, groups=b.match_id)):
        wpm = HGC(random_state=f, **WP_PARAMS).fit(Xs.iloc[tr], b.bat_win.iloc[tr])
        tl = tr[legal[tr]]
        om = HGC(random_state=f, **GB_PARAMS).fit(Xo.iloc[tl], b.oc.iloc[tl])
        tn = tr[nonwide[tr]]
        ms = {'bat_runs': HGR(random_state=f, **GB_PARAMS).fit(Xo.iloc[tn], b.bat_runs.iloc[tn]),
              'bat_out': HGC(random_state=f, **GB_PARAMS).fit(Xo.iloc[tn], b.bat_out.iloc[tn].astype(int)),
              'bowl_runs': HGR(random_state=f, **GB_PARAMS).fit(Xo.iloc[tr], b.bowl_runs.iloc[tr]),
              'bowl_wkt': HGC(random_state=f, **GB_PARAMS).fit(Xo.iloc[tr], b.bowl_wkt.iloc[tr].astype(int))}
        for k, mm in ms.items():
            E[k][te] = mm.predict(Xo.iloc[te]) if k.endswith('runs') else mm.predict_proba(Xo.iloc[te])[:, 1]
        d = b.iloc[te]
        args = lambda bl, w, r: (d.innings, bl, w, r, d.target, d.scope_i, d.venue_avg)
        base = wp_fun(wpm, *args(d.balls_left, d.wkts_before, d.runs_before))
        wp[te] = base
        P = om.predict_proba(Xo.iloc[te])                        # classes 0..6 (6 = wicket)
        li = np.zeros(len(te))
        for j, rv in enumerate(RUN_VALUES):
            nxt = wp_fun(wpm, *args(d.balls_left - 1, d.wkts_before, d.runs_before + rv))
            li += P[:, j] * np.abs(nxt - base)
        nxt_w = wp_fun(wpm, *args(d.balls_left - 1, d.wkts_before + 1, d.runs_before))
        li += P[:, 6] * np.abs(nxt_w - base)
        LI[te] = li
        if verbose: print(f'  fold {f}: WP log-loss {log_loss(b.bat_win.iloc[te], np.clip(base, 1e-4, 1 - 1e-4)):.4f}')
    b['wp'], b['LI'] = wp, LI
    for k, v in E.items(): b['E_' + k] = v
    ll = log_loss(b.bat_win, np.clip(b.wp, 1e-4, 1 - 1e-4)); base_ll = log_loss(b.bat_win, np.full(n, b.bat_win.mean()))
    metrics = dict(wp_logloss=ll, wp_baseline_logloss=base_ll, wp_brier=brier_score_loss(b.bat_win, b.wp))
    if verbose: print(f'  WP log-loss {ll:.4f} (coin-flip {base_ll:.4f}), Brier {metrics["wp_brier"]:.4f}')
    return b, metrics

def add_wicket_value(b, verbose=True):
    """Run cost of losing a wicket = E[remaining runs | after dot ball] - E[remaining runs | after wicket],
    from an out-of-fold expected-remaining-runs model fitted on first innings states."""
    b = b.copy()
    tot = b.groupby(['match_id', 'innings']).team_runs.transform('sum'); b['rem'] = tot - b.runs_before
    F = lambda bl, w, r, sc, ven: feats(np.ones(len(bl)), bl, w, r, np.nan, sc, ven)[['bl', 'w', 'r', 'crr', 'sc', 'ven']]
    ms = b.match_id.unique(); rng = np.random.RandomState(1)
    fold = pd.Series(rng.randint(0, 5, len(ms)), index=ms); b['fold'] = b.match_id.map(fold)
    wv = np.zeros(len(b))
    for f in range(5):
        tr = b[(b.innings == 1) & (b.fold != f)]
        m = HGR(max_iter=150, learning_rate=0.05, max_leaf_nodes=10, min_samples_leaf=300, l2_regularization=3, random_state=0)
        m.fit(F(tr.balls_left, tr.wkts_before, tr.runs_before, tr.scope_i, tr.venue_avg), tr.rem)
        te = b[b.fold == f]
        dot = m.predict(F(te.balls_left - 1, te.wkts_before, te.runs_before, te.scope_i, te.venue_avg))
        wk = m.predict(F(te.balls_left - 1, te.wkts_before + 1, te.runs_before, te.scope_i, te.venue_avg))
        wv[(b.fold == f).values] = dot - wk
    b['WV'] = np.clip(wv, 2, 40)
    if verbose: print('  wicket value by phase (runs):', b.groupby('phase', observed=True).WV.mean().round(1).to_dict())
    return b.drop(columns=['fold', 'rem'])
