"""Experiment driver for TENURE (E1-E10). Results go to $TENURE_OUT (default ./results).
Set QUICK=1 for a short smoke run (Colab free tier)."""
import os, sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
import tenure_sim as T

OUT = os.environ.get('TENURE_OUT', '.')
QUICK = os.environ.get('QUICK', '0') == '1'
os.makedirs(OUT, exist_ok=True)
NS = 3 if QUICK else 30          # main seeds
NS2 = 2 if QUICK else 12         # sweep seeds
TEST = list(range(1, NS + 1))
TUNE = list(range(101, 101 + (2 if QUICK else 10)))
CPU = int(os.environ.get('NPROC', os.cpu_count() or 1))


def job(a):
    tag, m, seed, kw = a
    t = time.time()
    r = T.run_one(m, seed, **kw)
    r['seed'] = seed; r['tag'] = tag; r['kw'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
    r['wall_s'] = time.time() - t
    if tag not in ('E1',):
        r.pop('alive_trace', None)
    return r


def run(name, jobs):
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        print('skip', name, flush=True)
        return json.load(open(path))
    t = time.time()
    with Pool(CPU) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(path, 'w'))
    print(name, len(jobs), 'runs', round(time.time() - t), 's', flush=True)
    return res


def main():
    stages = sys.argv[1:] or ['tune', 'E1', 'E2', 'E3', 'E4', 'E5', 'E10']
    # ---- tuning of the fixed-tenure baseline on separate seeds
    if 'tune' in stages:
        grid = [10, 20, 40, 80, 160, 320]
        res = run('tune_fixedT', [('tune', 'FixedTenure-PQ', s, dict(fixedT=g)) for g in grid for s in TUNE])
        best = max(grid, key=lambda g: np.mean([r['FND'] for r in res if r['kw']['fixedT'] == g]))
        json.dump({'fixedT': best}, open(os.path.join(OUT, 'tuned.json'), 'w'))
    Tbest = json.load(open(os.path.join(OUT, 'tuned.json')))['fixedT'] if os.path.exists(os.path.join(OUT, 'tuned.json')) else 80
    fx = dict(fixedT=Tbest)
    if 'E1' in stages:
        jobs = []
        for scen in [False, True]:
            for m in T.METHODS:
                kw = dict(attack=scen); kw.update(fx if m == 'FixedTenure-PQ' else {})
                jobs += [('E1', m, s, kw) for s in TEST]
        run('E1_main', jobs)
    if 'E2' in stages:
        abl = ['tenure', 'bneck', 'ratchet', 'frag', 'admit', 'detect', 'trust']
        jobs = []
        for scen in [False, True]:
            for a in [None] + abl:
                if not scen and a in ('admit', 'detect', 'trust'):
                    continue
                kw = dict(attack=scen)
                if a: kw['ablate'] = (a,)
                jobs += [('E2', 'TENURE', s, kw) for s in TEST[:NS2 + 8]]
        run('E2_ablation', jobs)
    if 'E3' in stages:
        kap = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
        jobs = [('E3', 'TENURE', s, dict(kappa=k)) for k in kap for s in TEST[:NS2 + 8]]
        jobs += [('E3f', 'FixedTenure-PQ', s, dict(fixedT=g)) for g in [10, 20, 40, 80, 160, 320] for s in TEST[:NS2 + 8]]
        run('E3_tenure', jobs)
    if 'E4' in stages:
        jobs = []
        for rr in [0, 2, 4, 8, 16]:
            for m in ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ', 'PQ-LEACH']:
                kw = dict(attack=True, r_in=rr, r_out=rr); kw.update(fx if m == 'FixedTenure-PQ' else {})
                jobs += [('E4r', m, s, kw) for s in TEST[:NS2]]
        for fc in [0.0, 0.05, 0.10, 0.20, 0.30]:
            for m, ab in [('TENURE', None), ('TENURE', 'trust'), ('DEEC-PQ', None)]:
                kw = dict(attack=True, frac_comp=fc)
                if ab: kw['ablate'] = (ab,)
                jobs += [('E4c', m, s, kw) for s in TEST[:NS2]]
        run('E4_robust', jobs)
    if 'E5' in stages:
        jobs = []
        for N in [50, 100, 200]:
            for m in ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ']:
                kw = dict(N=N); kw.update(fx if m == 'FixedTenure-PQ' else {})
                jobs += [('E5n', m, s, kw) for s in TEST[:NS2]]
        for kem in ['512', '768', '1024']:
            for m in ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ']:
                kw = dict(kem=kem); kw.update(fx if m == 'FixedTenure-PQ' else {})
                jobs += [('E5k', m, s, kw) for s in TEST[:NS2]]
        for m in ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ']:
            kw = dict(bs='center'); kw.update(fx if m == 'FixedTenure-PQ' else {})
            jobs += [('E5b', m, s, kw) for s in TEST[:NS2]]
        run('E5_general', jobs)
    if 'E10' in stages:
        jobs = []
        for lg in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for m in ['TENURE', 'DEEC-PQ', 'PQ-LEACH']:
                jobs += [('E10', m, s, dict(legacy=lg)) for s in TEST[:NS2]]
        run('E10_legacy', jobs)


if __name__ == '__main__':
    main()
