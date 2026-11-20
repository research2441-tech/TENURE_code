"""E8: anytime-valid join-flood detection vs CUSUM and fixed-rate rules.
Monte-Carlo over independent identities; writes $TENURE_OUT/E8_detector.json"""
import os, json
import numpy as np

OUT = os.environ.get('TENURE_OUT', '.')
QUICK = os.environ.get('QUICK', '0') == '1'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
M = 2000 if QUICK else 20000            # identities per condition
H = 5000                                  # rounds
LAM0, LAMS, ALPHA = 0.05, np.array([0.5, 1.0, 2.0, 4.0]), 0.01


def eproc_alarm(K):
    """first alarm round per row for mixture Poisson e-process (inf if none)."""
    lr = (K[..., None] * np.log(LAMS / LAM0) - (LAMS - LAM0))     # M x H x L
    logW = np.cumsum(lr, axis=1)
    mx = logW.max(-1)
    mix = mx + np.log(np.exp(logW - mx[..., None]).mean(-1))
    hit = mix >= np.log(1 / ALPHA)
    return np.where(hit.any(1), hit.argmax(1), np.inf)


def cusum_alarm(K, h, lam1=1.0):
    s = np.zeros(K.shape[0]); first = np.full(K.shape[0], np.inf)
    inc = K * np.log(lam1 / LAM0) - (lam1 - LAM0)
    for t in range(K.shape[1]):
        s = np.maximum(0, s + inc[:, t])
        new = (s >= h) & ~np.isfinite(first)
        first[new] = t
    return first


def fixed_alarm(K, kmax=2):
    hit = K >= kmax
    return np.where(hit.any(1), hit.argmax(1), np.inf)


def window_alarm(K, w=20, c=3):
    cs = np.cumsum(K, 1)
    win = cs - np.concatenate([np.zeros((K.shape[0], w)), cs[:, :-w]], 1)
    hit = win >= c
    return np.where(hit.any(1), hit.argmax(1), np.inf)


def null_counts(kind, lam, m):
    if kind == 'poisson':
        return rng.poisson(lam, size=(m, H))
    # over-dispersed: negative binomial with same mean, dispersion r=0.2
    r = 0.2; p = r / (r + lam)
    return rng.negative_binomial(r, p, size=(m, H))


def chunks(fn, K, *a, step=2000):
    return np.concatenate([fn(K[i:i + step], *a) for i in range(0, K.shape[0], step)])


def main():
    res = {'alpha': ALPHA, 'lam0': LAM0, 'H': H, 'M': M}
    # calibrate CUSUM threshold for 1% false alarms within 500 rounds under Poisson(lam0)
    Kc = null_counts('poisson', LAM0, M // 2)
    hs = np.linspace(2, 9, 29); best = None
    for h in hs:
        a = chunks(cusum_alarm, Kc[:, :500], h)
        if np.mean(a < 500) <= 0.01:
            best = float(h); break
    res['cusum_h'] = best
    horizons = [100, 500, 1000, 2000, 5000]
    far = {}
    for kind, lam in [('poisson', 0.02), ('poisson', 0.05), ('negbin', 0.05)]:
        K = null_counts(kind, lam, M)
        al = {'e-process': chunks(eproc_alarm, K, step=500), 'CUSUM': chunks(cusum_alarm, K, best),
              'fixed k>=2': fixed_alarm(K), 'window 3/20': window_alarm(K)}
        far[f'{kind}_{lam}'] = {m: [float(np.mean(a < t)) for t in horizons] for m, a in al.items()}
    res['far'] = far; res['horizons'] = horizons
    # detection delay for flooding identity (attack starts at t=0)
    rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    dd = {m: [] for m in ['e-process', 'CUSUM', 'fixed k>=2', 'window 3/20']}
    for r in rates:
        K = rng.poisson(r, size=(M // 4, 400))
        al = {'e-process': chunks(eproc_alarm, K, step=500), 'CUSUM': chunks(cusum_alarm, K, best),
              'fixed k>=2': fixed_alarm(K), 'window 3/20': window_alarm(K)}
        for m, a in al.items():
            a = np.where(np.isfinite(a), a + 1, 400)
            dd[m].append([float(np.mean(a)), float(np.percentile(a, 95))])
    res['rates'] = rates; res['delay'] = dd
    json.dump(res, open(os.path.join(OUT, 'E8_detector.json'), 'w'), indent=1)
    print(json.dumps(res['far'], indent=1)); print(res['cusum_h'])


if __name__ == '__main__':
    main()
