"""Optional step: figures used in the report."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_charts(b, q, full, avail, rows, out_dir):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for i in (1, 2):
        ax[0].plot(range(1, 21), b[b.innings == i].groupby('overid').LI.mean() * 100, marker='o', ms=3, label=f'Innings {i}')
    ax[0].axhline(q * 100, ls='--', c='grey', lw=.8); ax[0].set_xlabel('Over'); ax[0].set_ylabel('Mean win-prob swing per ball (%)'); ax[0].legend()
    ax[1].hist(b.LI * 100, bins=80, range=(0, 10), color='#2a6fdb'); ax[1].axvline(q * 100, c='r', ls='--'); ax[1].set_xlabel('Leverage per ball (%)')
    plt.tight_layout(); plt.savefig(out_dir / 'fig_pressure.png', dpi=170); plt.close()
    p, y = b.wp.values, b.bat_win.values; idx = np.digitize(p, np.linspace(0, 1, 11)[1:-1])
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.plot([0, 1], [0, 1], '--', c='grey')
    ax.plot([p[idx == i].mean() for i in range(10)], [y[idx == i].mean() for i in range(10)], 'o-')
    ax.set_xlabel('Predicted win probability'); ax.set_ylabel('Actual win rate'); plt.tight_layout(); plt.savefig(out_dir / 'fig_calibration.png', dpi=170); plt.close()
    for k, c in [('BAT', '#c2185b'), ('BOWL', '#1565c0')]:
        r = rows[k].head(10); fig, ax = plt.subplots(figsize=(6.2, 3.4)); yy = np.arange(10)[::-1]
        ax.barh(yy, r.PRS, color=c); ax.set_yticks(yy); ax.set_yticklabels([p_ + (' *' if cd else '') for p_, cd in zip(r.Player, r.Conditional)])
        for y_, v, t in zip(yy, r.PRS, r.Top10_boot_pct): ax.text(v + .03, y_, f'top-10 in {t:.0f}% of bootstraps', va='center', fontsize=7)
        ax.set_xlim(0, r.PRS.max() * 1.6); ax.set_xlabel('Pressure Reliability Score'); plt.tight_layout()
        plt.savefig(out_dir / f'fig_top10_{k.lower()}.png', dpi=170); plt.close()
