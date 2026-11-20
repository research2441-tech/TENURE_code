"""Turn raw results into figures, tables and the numeric dictionary R used in the manuscript."""
import os, json, collections, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.environ.get('TENURE_OUT', '.')
FIG = os.environ.get('TENURE_FIG', '.')
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(7)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'legend.frameon': False, 'mathtext.fontset': 'dejavuserif'})
M = ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ', 'PQ-LEACH', 'pqDTLS-LEACH', 'MH-RSA']
STY = {'TENURE': ('0.0', '-', 'o'), 'FixedTenure-PQ': ('0.25', '--', 's'), 'DEEC-PQ': ('0.4', '-.', '^'),
       'PQ-LEACH': ('0.5', ':', 'v'), 'pqDTLS-LEACH': ('0.62', (0, (5, 1.5, 1, 1.5)), 'D'), 'MH-RSA': ('0.75', (0, (1, 1)), 'x')}


def load(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def cliff(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return float(((x[:, None] > y[None]).sum() - (x[:, None] < y[None]).sum()) / (len(x) * len(y)))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def by(res, keyf):
    d = collections.defaultdict(list)
    for r in res:
        d[keyf(r)].append(r)
    for k in d:
        d[k].sort(key=lambda r: r['seed'])
    return d


def fmt(x, nd=0):
    return f'{x:,.{nd}f}' if np.isfinite(x) else '–'


R = {}
if os.path.exists(os.path.join(RES, 'tuned.json')):
    R['Tfix'] = json.load(open(os.path.join(RES, 'tuned.json')))['fixedT']

# ------------------------------------------------------------ E1 main
E1 = load('E1_main')
TAB = {}
if E1:
    g = by(E1, lambda r: (r['method'], r['kw']['attack']))
    rows = {}
    for att in [False, True]:
        sc = 'atk' if att else 'ben'
        for m in M:
            rs = g[(m, att)]
            d = {k: np.array([r[k] for r in rs], float) for k in
                 ['FND', 'SL', 'PDR', 'J_per_reading', 'hs_share', 'delay_ms', 'mean_tenure', 'e_total', 'e_hs',
                  'e_cpu', 'e_data', 'e_sleep', 'e_attack', 'e_ctrl', 'rotations']}
            for k, v in d.items():
                R[f'{sc}_{m}_{k}'] = float(np.nanmean(v)); R[f'{sc}_{m}_{k}_sd'] = float(np.nanstd(v, ddof=1))
            for k in ['FND', 'SL']:
                lo, hi = boot_ci(d[k]); R[f'{sc}_{m}_{k}_lo'] = lo; R[f'{sc}_{m}_{k}_hi'] = hi
            R[f'{sc}_{m}_FND_cens'] = float(np.mean(d['FND'] >= d['SL']))
            rows[(sc, m)] = d
        # Friedman and pairwise
        for k in ['FND', 'SL', 'PDR']:
            mat = np.array([rows[(sc, m)][k] for m in M])
            R[f'{sc}_friedman_{k}'] = float(stats.friedmanchisquare(*mat).pvalue)
            ps, ds = [], []
            for m in M[1:]:
                a, b = rows[(sc, 'TENURE')][k], rows[(sc, m)][k]
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = 1.0
                ps.append(p); ds.append(cliff(a, b))
            for m, p, dd in zip(M[1:], holm(ps), ds):
                R[f'{sc}_p_{k}_{m}'] = float(p); R[f'{sc}_d_{k}_{m}'] = dd
    TAB['main'] = rows
    ta = [r for r in E1 if r['method'] == 'TENURE' and r['kw']['attack']]
    R['atk_flagged'] = float(np.mean([r['detected'] for r in ta])); R['atk_insiders'] = float(np.mean([r['n_insiders'] for r in ta]))
    R['atk_false_alarms'] = int(sum(r['false_alarms'] for r in ta)); R['atk_det_delay'] = float(np.nanmean([r['det_delay'] for r in ta]))
    # survival figure
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True)
    for ax, att, ttl in zip(axs, [False, True], ['(a) benign', '(b) under attack']):
        for m in M:
            tr = [np.array(r['alive_trace'], float) for r in g[(m, att)]]
            n0 = np.array([t[0] for t in tr])
            L = max(len(t) for t in tr)
            Y = np.full((len(tr), L), np.nan)
            for i, t in enumerate(tr):
                Y[i, :len(t)] = t / n0[i]
            act = np.isfinite(Y).mean(0)
            med = np.nanmedian(Y, 0); x = np.arange(1, L + 1)
            ok = act >= 0.5
            c, ls, mk = STY[m]
            ax.plot(x[ok], med[ok], color=c, ls=ls, lw=1.3, label=m)
            ax.plot(x[ok][-1], med[ok][-1], marker=mk, color=c, ms=4, mfc='white', mew=1.0)
        ax.set_xscale('log'); ax.set_xlim(8, 5000); ax.set_ylim(0.45, 1.02)
        ax.set_xlabel('Round (log scale)'); ax.set_title(ttl, fontsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel('Fraction of honest nodes alive')
    axs[1].annotate('service ends (PDR < 0.5)\nfor DEEC-PQ, PQ-LEACH,\npqDTLS-LEACH', xy=(52, 0.995), xytext=(90, 0.78),
                    fontsize=6.6, ha='left', arrowprops=dict(arrowstyle='->', lw=0.7, color='0.2'))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, fontsize=7, bbox_to_anchor=(0.5, 1.08), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_survival.png'), dpi=400, bbox_inches='tight'); plt.close(fig)
    # energy breakdown
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
    comps = [('e_sleep', 'sleep', ''), ('e_data', 'data forwarding', '////'), ('e_ctrl', 'control', '....'),
             ('e_hs', 'key-establishment radio', 'xxxx'), ('e_cpu', 'cryptographic CPU', '\\\\\\\\'),
             ('e_attack', 'attack-induced', '----')]
    shades = ['1.0', '0.85', '0.7', '0.45', '0.3', '0.6']
    for ax, sc, ttl in zip(axs, ['ben', 'atk'], ['(a) benign', '(b) under attack']):
        y = np.arange(len(M))[::-1]
        left = np.zeros(len(M))
        for (k, lab, hat), sh in zip(comps, shades):
            v = np.array([R[f'{sc}_{m}_{k}'] / R[f'{sc}_{m}_e_total'] for m in M]) * 100
            ax.barh(y, v, left=left, color=sh, edgecolor='0.1', hatch=hat, lw=0.5, height=0.62, label=lab)
            left += v
        ax.set_yticks(y); ax.set_yticklabels(M, fontsize=7)
        ax.set_xlim(0, 100); ax.set_xlabel('Share of consumed energy (%)'); ax.set_title(ttl, fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, fontsize=6.6, bbox_to_anchor=(0.5, 1.09), handlelength=2.0, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_energy.png'), dpi=400, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------ E2 ablation
E2 = load('E2_ablation')
if E2:
    g = by(E2, lambda r: (r['kw']['attack'], tuple(r['kw'].get('ablate', []))))
    for (att, ab), rs in g.items():
        sc = 'atk' if att else 'ben'; nm = ab[0] if ab else 'full'
        for k in ['FND', 'SL', 'PDR', 'hs_share', 'e_attack', 'mean_tenure', 'J_per_reading']:
            v = np.array([r[k] for r in rs], float)
            R[f'abl_{sc}_{nm}_{k}'] = float(np.nanmean(v)); R[f'abl_{sc}_{nm}_{k}_sd'] = float(np.nanstd(v, ddof=1))
        full = g[(att, ())]
        if ab:
            for k in ['FND', 'SL', 'PDR']:
                a = np.array([r[k] for r in full], float); b = np.array([r[k] for r in rs], float)
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = 1.0
                R[f'abl_{sc}_{nm}_{k}_p'] = float(p); R[f'abl_{sc}_{nm}_{k}_d'] = cliff(a, b)

# ------------------------------------------------------------ E3 tenure sensitivity
E3 = load('E3_tenure')
if E3:
    gk = by([r for r in E3 if r['tag'] == 'E3'], lambda r: r['kw']['kappa'])
    gf = by([r for r in E3 if r['tag'] == 'E3f'], lambda r: r['kw']['fixedT'])
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.55))
    ks = sorted(gk); Ts = sorted(gf)
    tenk = np.array([np.nanmean([r['mean_tenure'] for r in gk[k]]) for k in ks])
    tenf = np.array([np.nanmean([r['mean_tenure'] for r in gf[t]]) for t in Ts])
    R['kap_tenure_rounds'] = {str(k): float(v) for k, v in zip(ks, tenk)}
    R['fixT_tenure_rounds'] = {str(t): float(v) for t, v in zip(Ts, tenf)}
    for ax, key, lab in [(axs[0], 'FND', 'First honest node death (round)'), (axs[1], 'SL', 'Service lifetime (round)')]:
        mu = np.array([np.mean([r[key] for r in gk[k]]) for k in ks])
        ci = np.array([boot_ci([r[key] for r in gk[k]]) for k in ks])
        muf = np.array([np.mean([r[key] for r in gf[t]]) for t in Ts])
        cif = np.array([boot_ci([r[key] for r in gf[t]]) for t in Ts])
        for k, v in zip(ks, mu):
            R[f'kap_{k}_{key}'] = float(v)
        for t, v in zip(Ts, muf):
            R[f'fixT_{t}_{key}'] = float(v)
        x = np.array(ks)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color='0.85', lw=0)
        ax.plot(x, mu, color='0.0', marker='o', ms=3.8, lw=1.3, label=r'TENURE (95 % CI shaded)')
        Tb = R.get('Tfix', 320)
        rf = [r[key] for r in gf[Tb]]; m0 = np.mean(rf); c0 = boot_ci(rf)
        ax.axhline(m0, color='0.35', ls='--', lw=1.0, label=f'FixedTenure-PQ, tuned $T$={Tb}')
        ax.axhspan(c0[0], c0[1], color='0.35', alpha=0.10, lw=0)
        ax.axvline(1.0, color='0.3', lw=0.6, ls=':')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(ks)); ax.xaxis.set_major_formatter(FixedFormatter([f'{k:g}' for k in ks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.2, 5)
        ax.set_xlabel(r'Tenure gain $\kappa$ in (#eq:stop)'.replace('(#eq:stop)', 'the stopping rule')); ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=6.4, loc='lower right')
    axs[0].set_title('(a)', fontsize=8); axs[1].set_title('(b)', fontsize=8)
    axs[0].set_ylim(850, 1250); axs[1].set_ylim(2150, 2680)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, 'fig5_tenure.png'), dpi=400, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------ E4 robustness
E4 = load('E4_robust')
if E4:
    gr = by([r for r in E4 if r['tag'] == 'E4r'], lambda r: (r['method'], r['kw']['r_in']))
    gc = by([r for r in E4 if r['tag'] == 'E4c'], lambda r: (r['method'] + ('-trust' if r['kw'].get('ablate') else ''), r['kw']['frac_comp']))
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45))
    for m in ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ', 'PQ-LEACH']:
        rr = sorted({k[1] for k in gr if k[0] == m})
        for x in rr:
            R[f'rob_{m}_{x}_SL'] = float(np.mean([r['SL'] for r in gr[(m, x)]]))
            R[f'rob_{m}_{x}_PDR'] = float(np.mean([r['PDR'] for r in gr[(m, x)]]))
            R[f'rob_{m}_{x}_FND'] = float(np.mean([r['FND'] for r in gr[(m, x)]]))
            R[f'rob_{m}_{x}_att'] = float(np.mean([r['e_attack'] / r['e_total'] for r in gr[(m, x)]]))
    for m, mk, c in [('TENURE', 'o', '0.0'), ('FixedTenure-PQ', 's', '0.5')]:
        rr = sorted({k[1] for k in gr if k[0] == m})
        for key, ls, nm in [('SL', '-', 'SL'), ('FND', '--', 'FND')]:
            mu = np.array([np.mean([r[key] for r in gr[(m, x)]]) for x in rr])
            ci = np.array([boot_ci([r[key] for r in gr[(m, x)]]) for x in rr])
            axs[0].errorbar(np.array(rr) + (0.15 if m != 'TENURE' else 0), mu, yerr=[mu - ci[:, 0], ci[:, 1] - mu],
                            color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, capsize=2, mfc='white', label=f'{m}, {nm}')
    axs[0].set_xlabel('Join-flood rate per attacker (requests/round)'); axs[0].set_ylabel('Lifetime (round)')
    axs[0].set_ylim(0, 2900); axs[0].grid(alpha=0.25, lw=0.4); axs[0].legend(fontsize=6.4, loc='center right', ncol=1)
    lab = {'TENURE': ('TENURE', '0.0', '-', 'o'), 'TENURE-trust': ('TENURE without trust', '0.35', '--', 's'),
           'DEEC-PQ': ('DEEC-PQ', '0.55', '-.', '^')}
    for m, (nm, c, ls, mk) in lab.items():
        fc = sorted({k[1] for k in gc if k[0] == m})
        mu = [np.mean([r['PDR'] for r in gc[(m, x)]]) for x in fc]
        for x, v in zip(fc, mu):
            R[f'ins_{m}_{x}_PDR'] = float(v)
            R[f'ins_{m}_{x}_SL'] = float(np.mean([r['SL'] for r in gc[(m, x)]]))
        axs[1].plot(np.array(fc) * 100, mu, color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc='white', label=nm)
    axs[1].set_xlabel('Captured insiders (% of nodes)'); axs[1].set_ylabel('Packet delivery ratio')
    axs[1].set_ylim(0, 1.02); axs[1].grid(alpha=0.25, lw=0.4); axs[1].legend(fontsize=6.6, loc='lower left')
    for ax, t in zip(axs, ['(a)', '(b)']):
        ax.set_title(t, fontsize=8)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, 'fig7_robust.png'), dpi=400, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------ E8 detector
E8 = load('E8_detector')
if E8:
    R['E8'] = E8
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45))
    hz = E8['horizons']
    st = {'e-process': ('0.0', '-', 'o'), 'CUSUM': ('0.35', '--', 's'), 'fixed k>=2': ('0.55', '-.', '^'),
          'window 3/20': ('0.7', ':', 'D')}
    names = {'e-process': 'mixture e-process', 'CUSUM': 'CUSUM', 'fixed k>=2': r'fixed rule $k\geq2$',
             'window 3/20': '3 in 20 rounds'}
    for cond, ls_ in [('poisson_0.05', '-'), ('negbin_0.05', None)]:
        pass
    for m, (c, ls, mk) in st.items():
        y = np.maximum(np.array(E8['far']['poisson_0.05'][m]), 1e-4)
        axs[0].plot(hz, y, color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc='white', label=names[m])
        y2 = np.maximum(np.array(E8['far']['negbin_0.05'][m]), 1e-4)
        axs[0].plot(hz, y2, color=c, ls=ls, marker=mk, ms=3.0, lw=0.7, alpha=0.55)
    axs[0].axhline(E8['alpha'], color='0.2', lw=0.7, ls=':')
    axs[0].text(110, E8['alpha'] * 1.25, r'$\alpha=0.01$', fontsize=6.6)
    axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_ylim(1e-4, 1.6)
    axs[0].set_xlabel('Monitoring horizon (rounds)'); axs[0].set_ylabel('Probability of flagging an honest node')
    axs[0].grid(alpha=0.25, lw=0.4)
    for m, (c, ls, mk) in st.items():
        d = np.array(E8['delay'][m])
        axs[1].plot(E8['rates'], d[:, 0], color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc='white', label=names[m])
    axs[1].set_xscale('log'); axs[1].set_yscale('log')
    axs[1].set_xlabel('Flooding rate (requests/round)'); axs[1].set_ylabel('Mean detection delay (rounds)')
    axs[1].grid(alpha=0.25, lw=0.4)
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    axs[1].xaxis.set_major_locator(FixedLocator(E8['rates'])); axs[1].xaxis.set_major_formatter(FixedFormatter([f'{r:g}' for r in E8['rates']]))
    axs[1].xaxis.set_minor_locator(NullLocator())
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, 1.07), handlelength=2.6, columnspacing=1.5)
    axs[0].set_title('(a) honest identity: Poisson (bold) and over-dispersed (faint)', fontsize=7.4)
    axs[1].set_title('(b) flooding identity', fontsize=8)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, 'fig6_detector.png'), dpi=400, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------ E5 generalisation
E5 = load('E5_general')
if E5:
    for tag, kf in [('E5n', lambda r: r['kw']['N']), ('E5k', lambda r: r['kw']['kem']), ('E5b', lambda r: 'center')]:
        g = by([r for r in E5 if r['tag'] == tag], lambda r: (r['method'], kf(r)))
        for (m, x), rs in g.items():
            for k in ['FND', 'SL', 'PDR', 'hs_share', 'mean_tenure']:
                R[f'gen_{tag}_{m}_{x}_{k}'] = float(np.nanmean([r[k] for r in rs]))
                R[f'gen_{tag}_{m}_{x}_{k}_sd'] = float(np.nanstd([r[k] for r in rs], ddof=1))
            if m != 'TENURE':
                t = [r['FND'] for r in g[('TENURE', x)]]; o = [r['FND'] for r in rs]
                R[f'gen_{tag}_{m}_{x}_dFND'] = cliff(t, o)

# ------------------------------------------------------------ E10 legacy
E10 = load('E10_legacy')
if E10:
    g = by(E10, lambda r: (r['method'], r['kw']['legacy']))
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35))
    for m in ['TENURE', 'DEEC-PQ', 'PQ-LEACH']:
        lg = sorted({k[1] for k in g if k[0] == m})
        ex = [np.mean([r['exposure'] for r in g[(m, x)]]) * 100 for x in lg]
        sl = [np.mean([r['SL'] for r in g[(m, x)]]) for x in lg]
        for x, a, b in zip(lg, ex, sl):
            R[f'leg_{m}_{x}_exp'] = float(a); R[f'leg_{m}_{x}_SL'] = float(b)
        c, ls, mk = STY[m]
        axs[0].plot(np.array(lg) * 100, ex, color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc='white', label=m)
        axs[1].plot(np.array(lg) * 100, sl, color=c, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc='white', label=m)
    axs[0].plot([0, 40], [0, 40], color='0.6', lw=0.6, ls=':')
    axs[0].text(24, 13, 'dotted: exposure = legacy share', fontsize=6.2, color='0.35')
    axs[0].set_xlabel('Legacy (X25519-only) nodes (%)'); axs[0].set_ylabel('Exposed delivered readings (%)')
    axs[1].set_xlabel('Legacy (X25519-only) nodes (%)'); axs[1].set_ylabel('Service lifetime (round)')
    axs[1].set_yscale('log')
    axs[1].set_ylim(300, 14000)
    for ax, t, loc in zip(axs, ['(a)', '(b)'], ['upper left', 'upper center']):
        ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=6.6, loc=loc, ncol=1 if loc == 'upper left' else 3); ax.set_title(t, fontsize=8)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, 'fig8_legacy.png'), dpi=400, bbox_inches='tight'); plt.close(fig)

json.dump(R, open(os.path.join(RES, 'analysis.json'), 'w'), indent=1, default=float)
print('analysis keys', len(R))
