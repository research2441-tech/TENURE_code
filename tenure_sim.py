"""
TENURE simulator: handshake-aware co-optimisation of cluster-head tenure,
rekeying and trust-weighted routing for post-quantum secure WSNs.

Round-based discrete-event simulation of an IEEE 802.15.4 clustered WSN.
Energy constants: nRF52840 product specification (3 V, DC/DC).
Cryptographic cycle counts: pqm4 (Cortex-M4), Lenngren 2025 (X25519),
CycloneCRYPTO 2.5.0 on Cortex-M4 (RSA-2048), ascon-c 32-bit ARMv7 proxy (Ascon).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from math import comb, ceil, sqrt, log

# ----------------------------------------------------------------- constants
V = 3.0
I_TX, I_RX = 4.8e-3, 4.6e-3            # A (nRF52840, 0 dBm TX, RX)
T_BYTE = 32e-6                          # s per byte at 250 kb/s
E_TX_B = V * I_TX * T_BYTE              # J per on-air byte
E_RX_B = V * I_RX * T_BYTE
E_CYC = V * 52e-6 * 64 / 64e6           # J per CPU cycle (52 uA/MHz @64 MHz)
P_SLEEP = V * 1.5e-6                    # W, System ON + RTC wake
ROUND_S = 60.0
GUARD_S = 0.5e-3                        # guard listening per scheduled RX
PHY_OVH = 6                             # preamble+SFD+length
MAC_OVH = 17                            # MAC header + FCS (short addresses)
FRAG_OVH = 5                            # 6LoWPAN fragmentation header
FRAG_PAY = 127 - MAC_OVH - FRAG_OVH     # payload bytes per fragment (105)
ACK_AIR = 5 + PHY_OVH
MAX_RETX = 3                            # macMaxFrameRetries
ATTEMPT_OVH_S = 2.3e-3                  # backoff+CCA+turnaround+ACK wait

CYC = {  # cycles on Cortex-M4
    'kem512_kg': 392224, 'kem512_enc': 392864, 'kem512_dec': 430202,
    'kem768_kg': 644195, 'kem768_enc': 664654, 'kem768_dec': 714194,
    'dsa44_sign': 3943121, 'dsa44_vrfy': 1421623,
    'x25519': 441116,
    'rsa_priv': 820.2e-3 * 120e6, 'rsa_pub': 91.7e-3 * 120e6,
    'ascon_aead_B': 55.5, 'ascon_xof_B': 138.1,
}
SZ = {'ek512': 800, 'ct512': 768, 'ct768': 1088, 'tag': 16,
      'dsa_pk': 1312, 'dsa_sig': 2420, 'rsa_pk': 259, 'rsa_ct': 256,
      'x_pub': 32, 'reading': 36, 'beacon': 20, 'adv': 20, 'join': 20}


def frags(s):
    return 1 if s <= 127 - MAC_OVH else int(ceil(s / FRAG_PAY))


def air_bytes(s):
    f = frags(s)
    if f == 1:
        return s + MAC_OVH + PHY_OVH, 1
    return s + f * (MAC_OVH + FRAG_OVH + PHY_OVH), f


# ------------------------------------------------------------ link model
def oqpsk_ber(snr_lin):
    """IEEE 802.15.4 O-QPSK/DSSS bit error rate (2.4 GHz PHY)."""
    s = np.zeros_like(snr_lin)
    for k in range(2, 17):
        s += (-1) ** k * comb(16, k) * np.exp(20.0 * snr_lin * (1.0 / k - 1.0))
    return np.clip(8.0 / 15.0 / 16.0 * s, 0.0, 0.5)


class Topology:
    def __init__(self, N, rng, side=None, pl0=40.0, eta=3.5, sigma=4.0,
                 pt=0.0, noise=-105.0, bs='edge'):
        self.N = N
        side = side or 200.0 * sqrt(N / 100.0)
        self.side = side
        pos = rng.uniform(0, side, size=(N, 2))
        if bs == 'edge':
            b = np.array([[side / 2, side + 10.0]])
        else:
            b = np.array([[side / 2, side / 2]])
        self.pos = np.vstack([pos, b])          # BS index N
        d = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        np.fill_diagonal(d, 1.0)
        d = np.maximum(d, 1.0)
        self.d = d
        X = rng.normal(0, sigma, size=(N + 1, N + 1))
        X = np.triu(X, 1); X = X + X.T
        pl = pl0 + 10 * eta * np.log10(d) + X
        snr_db = pt - pl - noise
        self.snr_db = snr_db
        ber = oqpsk_ber(10 ** (snr_db / 10.0))
        np.fill_diagonal(ber, 0.5)
        self.ber = ber
        self._cache = {}
        self.p_ack = (1 - ber) ** (8 * ACK_AIR)
        self.link = self.prr(127 + PHY_OVH) >= 0.10   # usable link
        np.fill_diagonal(self.link, False)

    def prr(self, air):
        key = round(float(air), 3)
        c = self._cache.get(key)
        if c is None:
            c = (1.0 - self.ber) ** (8 * key)
            self._cache[key] = c
        return c


# ---------------------------------------------------------- message costs
def hop_stats(topo, u, v, s):
    """Expected energies/latency and success probability for an s-byte
    message over hop u->v with per-fragment ARQ (arrays allowed)."""
    tot, f = air_bytes(s)
    per = tot / f
    p = topo.prr(per)[u, v] * topo.p_ack[u, v]
    p = np.maximum(p, 1e-6)
    fail = (1 - p) ** (MAX_RETX + 1)
    q = 1 - fail                                  # fragment success prob
    att = q / p                                   # expected attempts/fragment
    mf = np.where(q < 1 - 1e-12, (1 - q ** f) / np.maximum(1 - q, 1e-12), f)
    e_tx = mf * att * (per * E_TX_B + ACK_AIR * E_RX_B)
    e_rx = mf * att * (per * E_RX_B) + mf * att * p * ACK_AIR * E_TX_B
    lat = mf * att * (per * T_BYTE + ATTEMPT_OVH_S)
    return e_tx, e_rx, lat, q ** f, mf * att


# ------------------------------------------------------------- e-process
class EProcess:
    """Mixture Poisson likelihood-ratio e-process per identity.
    Null: join-request rate <= lam0 per round."""
    def __init__(self, n, lam0=0.05, lams=(0.5, 1.0, 2.0, 4.0), alpha=0.01):
        self.lam0 = lam0
        self.lams = np.array(lams)
        self.logW = np.zeros((n, len(lams)))
        self.thr = log(1.0 / alpha)
        self.flag = np.zeros(n, bool)

    def update(self, idx, k):
        if len(idx) == 0:
            return np.zeros(0, bool)
        lr = (k[:, None] * np.log(self.lams[None] / self.lam0)
              - (self.lams[None] - self.lam0))
        self.logW[idx] += lr
        m = self.logW[idx].max(1)
        mix = m + np.log(np.exp(self.logW[idx] - m[:, None]).mean(1))
        new = (mix >= self.thr) & ~self.flag[idx]
        self.flag[idx[new]] = True
        return new


# ------------------------------------------------------------- simulator
DEFAULT = dict(N=100, E0=1.0, p_ch=0.15, rounds_max=20000, stop='hnd',
               attack=False, frac_comp=0.10, drop_p=0.8, r_in=4, r_out=6,
               n_out=2, n_sybil=2, legacy=0.0, kem='512', kappa=1.0,
               fixedT=None, e_reserve=0.05, tau_min=0.6, gamma=0.98,
               eta_E=1.0, lam_T=2.0, alpha=0.01, lam0=0.05, bs='edge',
               legit_rejoin=0.02)

METHODS = ['TENURE', 'FixedTenure-PQ', 'DEEC-PQ', 'PQ-LEACH',
           'pqDTLS-LEACH', 'MH-RSA']


class Sim:
    def __init__(self, method, seed, **kw):
        self.cfg = dict(DEFAULT); self.cfg.update(kw)
        c = self.cfg
        self.method = method
        # component switches (ablations)
        self.use_tenure = method in ('TENURE', 'FixedTenure-PQ') or method.startswith('TENURE')
        self.abl = set(kw.get('ablate', ()))
        self.rng = np.random.default_rng(seed)
        self.topo = Topology(c['N'], np.random.default_rng(seed + 7919), bs=c['bs'])
        N = c['N']; self.N = N; self.BS = N
        self.E = np.full(N, c['E0'])
        self.alive = np.ones(N, bool)
        rng = np.random.default_rng(seed + 104729)
        nc = int(round(c['frac_comp'] * N)) if c['attack'] else 0
        self.comp = np.zeros(N, bool)
        self.comp[rng.choice(N, nc, replace=False)] = True
        self.legacy = np.zeros(N, bool)
        nl = int(round(c['legacy'] * N))
        if nl:
            cand = np.where(~self.comp)[0]
            self.legacy[rng.choice(cand, nl, replace=False)] = True
        # outsider attackers (unlimited energy)
        self.out_pos = rng.uniform(0, self.topo.side, size=(c['n_out'], 2)) if c['attack'] else np.zeros((0, 2))
        # trust (Beta) for TENURE
        self.ta = np.ones(N); self.tb = np.ones(N)
        self.ep = EProcess(N, lam0=c['lam0'], alpha=c['alpha'])
        self.blacklist = np.zeros(N, bool)
        self.false_alarm = 0; self.detect_round = {}
        # bookkeeping
        self.stats = dict(gen=0, dlv=0, e_hs=0.0, e_cpu=0.0, e_data=0.0,
                          e_sleep=0.0, e_attack=0.0, e_ctrl=0.0, lat=[],
                          rotations=0, exposed=0, dlv_total=0, tenures=[])
        self.alive_trace = []
        self.cluster_init = False
        self.ch_of = np.full(N, -1)       # member -> CH
        self.is_ch = np.zeros(N, bool)
        self.lifetime = {}

    # ------------------------------------------------------ energy helpers
    def charge(self, idx, e, kind):
        e = np.asarray(e, float)
        np.add.at(self.E, np.asarray(idx, int), -e)
        self.stats[kind] += float(np.sum(e))

    def cpu(self, idx, cycles):
        idx = np.atleast_1d(np.asarray(idx, int))
        self.charge(idx, np.full(len(idx), cycles * E_CYC), 'e_cpu')

    def trust(self):
        return self.ta / (self.ta + self.tb)

    # --------------------------------------------------- unicast & paths
    def unicast(self, u, v, s, kind, reliable=False, sample=True):
        """single hop u->v (v may be BS). returns success flag."""
        etx, erx, lat, ps, _ = hop_stats(self.topo, u, v, s)
        reps = 1.0
        if reliable:
            reps = (1 - (1 - ps) ** 3) / max(ps, 1e-6)
        if u < self.N: self.charge([u], [etx * reps], kind)
        if v < self.N: self.charge([v], [erx * reps], kind)
        ok = (self.rng.random() < (1 - (1 - ps) ** 3 if reliable else ps)) if sample else True
        return ok, lat

    def unicast_many(self, us, v, s, kind, reliable=False):
        us = np.asarray(us, int)
        if len(us) == 0:
            return np.zeros(0, bool)
        etx, erx, lat, ps, _ = hop_stats(self.topo, us, np.full(len(us), v), s)
        if reliable:
            reps = (1 - (1 - ps) ** 3) / np.maximum(ps, 1e-6)
            psucc = 1 - (1 - ps) ** 3
        else:
            reps = np.ones(len(us)); psucc = ps
        self.charge(us, etx * reps, kind)
        if v < self.N:
            self.charge([v], [float(np.sum(erx * reps))], kind)
        return self.rng.random(len(us)) < psucc

    def path_send(self, path, s, kind, drop_check=True, reliable=False):
        """send along node list path (last = BS). expected energy charged
        hop by hop weighted by reaching probability; sampled delivery."""
        reach = 1.0; lat = 0.0; delivered = True
        att_rep = 1.0
        for h in range(len(path) - 1):
            u, v = path[h], path[h + 1]
            etx, erx, l, ps, _ = hop_stats(self.topo, u, v, s)
            if reliable:
                rr = (1 - (1 - ps) ** 3) / max(ps, 1e-6)
                etx *= rr; erx *= rr; ps = 1 - (1 - ps) ** 3
            if u < self.N: self.charge([u], [etx * reach], kind)
            if v < self.N: self.charge([v], [erx * reach], kind)
            lat += l
            if delivered:
                if self.rng.random() > ps:
                    delivered = False
                elif drop_check and v < self.N and self.comp[v] and self.cfg['attack'] \
                        and self.rng.random() < self.cfg['drop_p'] and kind == 'e_data':
                    delivered = False
                    self._evidence(v, path[h], dropped=True)
                elif v < self.N and kind == 'e_data':
                    self._evidence(v, path[h], dropped=False)
            reach *= ps
        return delivered, lat

    def _evidence(self, v, u, dropped):
        if not self._tenure_like() or 'trust' in self.abl:
            return
        # watchdog at upstream u overhears v's forwarding with prob PRR(v->u)
        o = self.topo.prr(60)[v, u] if u < self.N else 0.9
        if self.rng.random() > o:
            return
        if dropped:
            if self.rng.random() < 0.9: self.tb[v] += 1
        else:
            if self.rng.random() < 0.05: self.tb[v] += 1
            else: self.ta[v] += 1

    def broadcast(self, c, members, s, kind):
        """CH c broadcasts s bytes to members (no ARQ, repeats until all)."""
        if len(members) == 0:
            return
        tot, f = air_bytes(s)
        per = tot / f
        q = np.maximum(self.topo.prr(per)[c, members] ** f, 1e-3)
        rep_m = np.minimum(1.0 / q, 4.0)
        self.charge([c], [tot * E_TX_B * rep_m.max()], kind)
        self.charge(members, tot * E_RX_B * rep_m, kind)

    def _tenure_like(self):
        return self.method.startswith('TENURE') or self.method == 'FixedTenure-PQ'

    # ------------------------------------------------------ routing
    def route_tree(self, size):
        """shortest-path predecessor tree towards BS for current alive set."""
        N = self.N; topo = self.topo
        al = np.append(self.alive, True)
        L = topo.link & al[:, None] & al[None, :]
        u, v = np.nonzero(L)
        m = self.method
        if self._tenure_like():
            _, _, _, ps, xm = hop_stats(topo, u, v, size)
            if 'frag' in self.abl:
                _, _, _, ps, xm = hop_stats(topo, u, v, 20)
            w = xm.astype(float)
            Ev = np.where(v < N, self.E[np.minimum(v, N - 1)], self.cfg['E0'])
            w = w * (1 + self.cfg['eta_E'] * (1 - Ev / self.cfg['E0']))
            if 'trust' not in self.abl:
                tv = np.where(v < N, self.trust()[np.minimum(v, N - 1)], 1.0)
                w = w + self.cfg['lam_T'] * (-np.log(np.maximum(tv, 1e-3)))
            if self.cfg['attack']:
                pass  # BS uses two-sided link reports; lies cannot lower costs
        elif m == 'MH-RSA':  # hop-count fallback (HHO overrides CH next hops)
            w = np.ones(len(u))
            if self.cfg['attack']:
                w = np.where((v < N) & self.comp[np.minimum(v, N - 1)], 1e-3, w)
        else:  # PQ-LEACH, pqDTLS-LEACH, DEEC-PQ: advertised ETX (RPL MRHOF-like)
            _, _, _, ps, xm = hop_stats(topo, u, v, 20)
            w = xm.astype(float)
            if self.cfg['attack']:
                w = np.where((v < N) & self.comp[np.minimum(v, N - 1)], 1e-3, w)
        G = csr_matrix((w, (v, u)), shape=(N + 1, N + 1))   # reversed graph
        dist, pred = dijkstra(G, indices=N, return_predecessors=True)
        self.pred = pred
        self.rdist = dist

    def path_from(self, c):
        p = [c]
        while p[-1] != self.BS:
            nx = self.pred[p[-1]]
            if nx < 0 or len(p) > self.N + 1:
                return None
            p.append(int(nx))
        return p

    # ------------------------------------------------------ handshakes
    def rekey_cluster(self, c, members, protocol):
        """cost of (re)establishing keys between CH c and members, and CH-BS."""
        st = self.stats
        n = len(members)
        leg_c = self.legacy[c]
        if protocol in ('pq', 'tenure'):
            tags = protocol == 'tenure'
            if leg_c:
                self.cpu(c, CYC['x25519'])
                self.broadcast(c, members, SZ['x_pub'] + (SZ['tag'] * n if tags else 0), 'e_hs')
            else:
                kg, ek = {'512': (CYC['kem512_kg'], SZ['ek512']), '768': (CYC['kem768_kg'], 1184),
                          '1024': (1020202, 1568)}[self.cfg['kem']]
                self.cpu(c, kg)
                self.broadcast(c, members, ek + (SZ['tag'] * n + 16 if tags else 0), 'e_hs')
            members = np.asarray(members, int)
            cl = members[self.legacy[members] | leg_c]
            pq = members[~(self.legacy[members] | leg_c)]
            if len(cl):
                self.cpu(cl, 2 * CYC['x25519']); self.cpu(c, len(cl) * CYC['x25519'])
                self.unicast_many(cl, c, SZ['x_pub'] + (SZ['tag'] if tags else 0), 'e_hs', reliable=True)
            if len(pq):
                kk = self.cfg['kem']
                enc, dec, ct = {'512': (CYC['kem512_enc'], CYC['kem512_dec'], SZ['ct512']),
                                '768': (CYC['kem768_enc'], CYC['kem768_dec'], SZ['ct768']),
                                '1024': (1037953, 1100982, 1568)}[kk]
                self.cpu(pq, enc); self.cpu(c, len(pq) * dec)
                self.unicast_many(pq, c, ct + (SZ['tag'] if tags else 0), 'e_hs', reliable=True)
            if tags and len(members):
                self.cpu(members, 2 * 64 * CYC['ascon_xof_B'])
            # CH <-> BS
            path = self.path_from(c)
            if path is not None:
                if leg_c:
                    self.cpu(c, 2 * CYC['x25519'])
                    self.path_send(path, SZ['x_pub'] + 16, 'e_hs', drop_check=False, reliable=True)
                else:
                    self.cpu(c, CYC['kem768_enc'])
                    self.path_send(path, SZ['ct768'] + 16, 'e_hs', drop_check=False, reliable=True)
                if tags:   # tickets BS -> CH (downlink along reverse path)
                    self.path_send(path[::-1], 16 * n + 32, 'e_hs', drop_check=False, reliable=True)
        elif protocol == 'dtls':
            self.rekey_cluster(c, members, 'pq')
            self.cpu(c, CYC['dsa44_sign'])
            self.broadcast(c, members, SZ['dsa_pk'] + 2 * SZ['dsa_sig'], 'e_hs')
            if len(members):
                self.cpu(np.asarray(members, int), 2 * CYC['dsa44_vrfy'])
        elif protocol == 'rsa':
            self.broadcast(c, members, SZ['rsa_pk'], 'e_hs')
            members = np.asarray(members, int)
            if len(members):
                self.cpu(members, CYC['rsa_pub']); self.cpu(c, len(members) * CYC['rsa_priv'])
                self.unicast_many(members, c, SZ['rsa_ct'], 'e_hs', reliable=True)
            path = self.path_from(c)
            if path is not None:
                self.cpu(c, CYC['rsa_pub'])
                self.path_send(path, SZ['rsa_ct'], 'e_hs', drop_check=False, reliable=True)

    # ------------------------------------------------------ attacks
    def floods(self, chs, protocol):
        """insider / outsider / Sybil join floods against cluster heads."""
        if not self.cfg['attack'] or len(chs) == 0:
            return
        c = self.cfg; topo = self.topo
        admission = self._tenure_like() and 'admit' not in self.abl
        ct_sz = SZ['ct512'] + (SZ['tag'] if admission else 0)
        tot, f = air_bytes(ct_sz)
        full_rx = tot * E_RX_B + f * ACK_AIR * E_TX_B
        one_rx = (tot / f) * E_RX_B
        dec = CYC['kem512_dec'] if protocol != 'rsa' else CYC['rsa_priv']
        if protocol == 'rsa':
            full_rx = air_bytes(SZ['rsa_ct'])[0] * E_RX_B
        ins = np.where(self.comp & self.alive & ~self.is_ch)[0]
        for a in ins:
            tgt = self.ch_of[a]
            if tgt < 0 or not self.alive[tgt]:
                continue
            k = self.rng.poisson(c['r_in'])
            if k == 0:
                continue
            self.charge([a], [k * tot * E_TX_B], 'e_attack')
            if admission and 'detect' not in self.abl:
                newf = self.ep.update(np.array([a]), np.array([k]))
                if newf.any():
                    self.detect_round.setdefault(a, self.r)
                    self.tb[a] += 5
            if admission and (self.ep.flag[a] and 'detect' not in self.abl):
                self.charge([tgt], [k * one_rx], 'e_attack')
            else:
                self.charge([tgt], [k * full_rx], 'e_attack')
                self.cpu(tgt, k * dec)
                self.stats['e_attack'] += k * dec * E_CYC; self.stats['e_cpu'] -= k * dec * E_CYC
        # Sybil identities from insiders: no ticket
        for a in ins:
            tgt = self.ch_of[a]
            if tgt < 0 or not self.alive[tgt]:
                continue
            k = c['n_sybil']
            if admission:
                self.charge([tgt], [k * one_rx], 'e_attack')
            else:
                self.charge([tgt], [k * full_rx], 'e_attack')
                self.cpu(tgt, k * dec)
        # outsiders: in range of CHs (prr from attacker position)
        for op in self.out_pos:
            dd = np.linalg.norm(self.topo.pos[chs] - op, axis=1)
            inr = chs[dd < 60.0]
            for tgt in inr:
                k = self.rng.poisson(c['r_out'])
                if admission:
                    self.charge([tgt], [k * one_rx], 'e_attack')
                else:
                    self.charge([tgt], [k * full_rx], 'e_attack')
                    self.cpu(tgt, k * dec)

    # ------------------------------------------------------ clustering
    def kmedoids_init(self):
        N = self.N; k = max(2, int(round(self.cfg['p_ch'] * N)))
        pr = self.topo.prr(60)[:N, :N]
        cost = np.where(self.topo.link[:N, :N], 1.0 / np.maximum(pr, 1e-3), 1e3)
        np.fill_diagonal(cost, 0)
        med = self.rng.choice(N, k, replace=False)
        for _ in range(20):
            lab = np.argmin(cost[:, med], axis=1)
            newm = med.copy()
            for j in range(k):
                mem = np.where(lab == j)[0]
                if len(mem):
                    newm[j] = mem[np.argmin(cost[np.ix_(mem, mem)].sum(1))]
            if np.array_equal(newm, med):
                break
            med = newm
        lab = np.argmin(cost[:, med], axis=1)
        self.cluster = lab
        self.ncl = k
        self.cl_ch = np.full(k, -1)
        self.cl_spent = np.zeros(k); self.cl_theta = np.full(k, np.inf)
        self.cl_start = np.zeros(k, int)
        self.cl_delta = np.full(k, np.nan); self.cl_P = np.full(k, np.nan)
        self.cl_H = np.full(k, np.nan)
        self.h_vec = np.zeros((k, N)); self.tau_cur = np.full(k, np.nan)
        self.p_node = np.full(N, np.nan); self.rk_round = np.zeros(N)
        self.cl_regime = np.zeros(k, int)
        self.cluster_init = True

    def elect_in_cluster(self, j, exclude=-1):
        N = self.N
        mem = np.where((self.cluster == j) & self.alive)[0]
        if len(mem) == 0:
            return -1
        E_claim = np.where(self.comp[mem], self.cfg['E0'], self.E[mem])
        ok = (self.E[mem] > self.cfg['e_reserve'] * self.cfg['E0']) & ~self.legacy[mem]
        if 'trust' not in self.abl:
            ok &= (self.trust()[mem] >= self.cfg['tau_min']) & ~self.blacklist[mem]
        ok &= mem != exclude
        if not ok.any():
            ok = self.E[mem] > 0
            if not ok.any():
                return -1
        pr = self.topo.prr(60)[np.ix_(mem, mem)]
        cen = pr.mean(1)
        sc = np.where(ok, E_claim / self.cfg['E0'] * cen, -1)
        return int(mem[np.argmax(sc)])

    def theta_star(self, j):
        """Tenure threshold: renewal (EOQ-type) optimum for the cluster,
        lifted to the max-min crossing with the bottleneck relay."""
        mem = np.where((self.cluster == j) & self.alive)[0]
        n = len(mem)
        d, P, H = self.cl_delta[j], self.cl_P[j], self.cl_H[j]
        if n < 2 or not np.isfinite(d) or not np.isfinite(P) or not np.isfinite(H) or d <= 0 or H <= 0:
            return np.inf
        A = float(self.E[mem].sum())
        B = (n - 1) * d
        tau_c = (H / P) * (sqrt(1.0 + A * P / (B * H)) - 1.0)
        tau = tau_c; self.cl_regime[j] = 0
        if 'bneck' not in self.abl:
            hj = self.h_vec[j]
            R = np.where((hj > 0) & self.alive & np.isfinite(self.p_node))[0]
            if len(R):
                tc = np.where(np.isfinite(self.tau_cur) & (self.tau_cur > 0), self.tau_cur, np.inf)
                rho = (self.h_vec[:, R] / tc[:, None]).sum(0)
                own = hj[R] / tc[j] if np.isfinite(tc[j]) else 0.0
                q = self.p_node[R] + rho - own
                h = hj[R]
                Lb = self.E[R] / (q + h / tau_c)
                b = int(np.argmin(Lb))
                Lc = (A - B * tau_c) / (P + H / tau_c)
                if Lb[b] < Lc:
                    Eb, qb, hb = self.E[R][b], q[b], h[b]
                    a2 = B * qb
                    a1 = -(A * qb - B * hb - Eb * P)
                    a0 = -(A * hb - Eb * H)
                    disc = a1 * a1 - 4 * a2 * a0
                    if a2 > 0 and disc >= 0:
                        tx = (-a1 + sqrt(disc)) / (2 * a2)
                        if tx > tau_c:
                            tau = min(tx, 0.9 * A / B); self.cl_regime[j] = 1
        self.tau_cur[j] = tau
        return self.cfg['kappa'] * tau * d

    # ------------------------------------------------------ main loop
    def run(self):
        c = self.cfg; N = self.N
        self.r = 0
        self.win = []
        self.n_hon = int((~self.comp).sum())
        prev_ch_set = None
        first_death = None
        while self.r < c['rounds_max']:
            r = self.r
            nalive = int(self.alive.sum())
            nhon = int((self.alive & ~self.comp).sum())
            self.alive_trace.append(nhon)
            if nhon <= self.n_hon // 2 or nalive < 3:
                break
            if len(self.win) >= 50 and sum(d for g, d in self.win) < 0.5 * sum(g for g, d in self.win):
                break
            E_before = self.E.copy()
            g0, d0 = self.stats['gen'], self.stats['dlv']
            if self._tenure_like():
                changed = self.round_tenure()
            else:
                changed = self.round_leach_family()
            # sleep / idle
            idx = np.where(self.alive)[0]
            self.charge(idx, np.full(len(idx), P_SLEEP * ROUND_S), 'e_sleep')
            # deaths
            dead = self.alive & (self.E <= 0)
            if dead.any():
                self.alive[dead] = False
                if first_death is None and (dead & ~self.comp).any():
                    first_death = r
            if self._tenure_like():
                self.post_round_tenure(E_before)
            self.win.append((self.stats['gen'] - g0, self.stats['dlv'] - d0))
            if len(self.win) > 50:
                self.win.pop(0)
            self.r += 1
        self.lifetime['FND'] = first_death if first_death is not None else self.r
        self.lifetime['SL'] = self.r
        return self.summary()

    # ---------------- LEACH family (PQ-LEACH, pqDTLS, DEEC-PQ, MH-RSA)
    def round_leach_family(self):
        c = self.cfg; N = self.N; m = self.method
        al = np.where(self.alive)[0]
        p = c['p_ch']
        if m == 'DEEC-PQ':
            Ec = np.where(self.comp, c['E0'], self.E)
            Ebar = Ec[al].mean()
            pi = np.clip(p * Ec[al] / max(Ebar, 1e-9), 0, 1)
            chs = al[self.rng.random(len(al)) < pi]
        elif m == 'MH-RSA':
            chs = self.swarm_clustering(al)
        else:
            if not hasattr(self, 'G_last'):
                self.G_last = np.full(N, -10 ** 9)
            epoch = int(round(1 / p))
            elig = al[(self.r - self.G_last[al]) >= epoch]
            T = p / (1 - p * (self.r % epoch))
            chs = elig[self.rng.random(len(elig)) < T]
            self.G_last[chs] = self.r
        if self.cfg['attack'] and m != 'MH-RSA':
            chs = np.union1d(chs, al[self.comp[al]])     # sinkhole self-election
        if len(chs) == 0:
            chs = al[[np.argmax(self.E[al])]]
        self.is_ch[:] = False; self.is_ch[chs] = True
        # membership by best link
        pr = self.topo.prr(60)
        mem = al[~self.is_ch[al]]
        best = chs[np.argmax(pr[np.ix_(mem, chs)], axis=1)] if len(mem) else np.array([], int)
        good = self.topo.link[mem, best] if len(mem) else np.array([], bool)
        self.ch_of[:] = -1
        self.ch_of[mem[good]] = best[good]
        self.orphans = mem[~good]
        self.ch_of[chs] = chs
        # control: ADV + JOIN + schedule
        for ch in chs:
            ms = np.where(self.ch_of == ch)[0]; ms = ms[ms != ch]
            self.broadcast(ch, al[al != ch][:0], SZ['adv'], 'e_ctrl')
        ADV = air_bytes(SZ['adv'])[0]
        self.charge(chs, np.full(len(chs), ADV * E_TX_B), 'e_ctrl')
        nonch = al[~self.is_ch[al]]
        # every non-CH hears ~ all CH advertisements in range
        heard = (pr[np.ix_(nonch, chs)] > 0.1).sum(1)
        self.charge(nonch, heard * ADV * E_RX_B, 'e_ctrl')
        # routing
        if m == 'MH-RSA':
            self.route_tree(64)
            self.hho_routes(chs)
        else:
            self.route_tree(64)
        proto = {'PQ-LEACH': 'pq', 'DEEC-PQ': 'pq', 'pqDTLS-LEACH': 'dtls', 'MH-RSA': 'rsa'}[m]
        for ch in chs:
            ms = np.where(self.ch_of == ch)[0]; ms = ms[ms != ch]
            self.unicast_many(ms, ch, SZ['join'], 'e_ctrl')
            self.broadcast(ch, ms, 20 + 2 * len(ms), 'e_ctrl')
            self.rekey_cluster(ch, ms, proto)
        self.stats['rotations'] += len(chs)
        self.floods(chs, proto)
        self.data_phase(chs)
        return True

    def swarm_clustering(self, al):
        """hybrid swarm (PSO with pheromone-guided global-best jumps) CH set."""
        c = self.cfg; k = max(2, int(round(c['p_ch'] * len(al))))
        pos = self.topo.pos[al]
        Ec = np.where(self.comp[al], c['E0'], self.E[al])
        P, it = 10, 10
        X = self.rng.uniform(0, self.topo.side, size=(P, k, 2)); Vv = np.zeros_like(X)

        def decode(x):
            dd = np.linalg.norm(pos[None, :, :] - x[:, None, :], axis=2)
            return np.unique(np.argmin(dd, axis=1))

        def fit(x):
            ids = decode(x)
            dd = np.linalg.norm(pos[:, None] - pos[ids][None], axis=2).min(1)
            return (dd ** 2).mean() / 100.0 + (c['E0'] / np.maximum(Ec[ids], 1e-6)).mean()
        F = np.array([fit(x) for x in X]); Pb = X.copy(); Pf = F.copy()
        g = Pb[np.argmin(Pf)].copy(); gf = Pf.min(); tau = 1.0
        for _ in range(it):
            r1, r2 = self.rng.random(X.shape), self.rng.random(X.shape)
            Vv = 0.7 * Vv + 1.5 * r1 * (Pb - X) + 1.5 * r2 * (g - X)
            X = np.clip(X + Vv, 0, self.topo.side)
            for i in range(P):
                F[i] = fit(X[i])
                if F[i] < Pf[i]:
                    Pf[i] = F[i]; Pb[i] = X[i].copy()
            # ant step: jump around global best scaled by pheromone level
            ant = np.clip(g + self.rng.normal(0, 0.05 * self.topo.side * tau, g.shape), 0, self.topo.side)
            fa = fit(ant)
            if fa < gf:
                g, gf = ant, fa; tau = tau + 0.01 * gf
            else:
                tau = 0.9 * tau
            if Pf.min() < gf:
                g = Pb[np.argmin(Pf)].copy(); gf = Pf.min()
        return al[decode(g)]

    def hho_routes(self, chs):
        """Harris hawks optimisation of CH next-hop (CH or BS) choices."""
        N = self.N; pr = self.topo.prr(60)
        dBS = self.topo.d[chs, N]
        cand = []
        for i, ch in enumerate(chs):
            cs = [N] if pr[ch, N] >= 0.3 else []
            cs += [int(o) for o in chs if o != ch and pr[ch, o] >= 0.3 and self.topo.d[o, N] < dBS[i]]
            cand.append(cs)
        k = len(chs)
        if k == 0:
            return
        Ec = np.where(self.comp, self.cfg['E0'], self.E)

        def decode(x):
            return [cand[i][min(int(x[i] * len(cand[i])), len(cand[i]) - 1)] if cand[i] else -1 for i in range(k)]

        def fit(x):
            nh = decode(x); f = 0.0
            for i, ch in enumerate(chs):
                if nh[i] < 0:
                    f += 50.0; continue
                f += self.topo.d[ch, nh[i]] ** 2 / 1e3 + (0 if nh[i] == N else self.cfg['E0'] / max(Ec[nh[i]], 1e-6))
            return f
        P, T = 8, 8
        X = self.rng.random((P, k)); F = np.array([fit(x) for x in X])
        rb = X[np.argmin(F)].copy(); fb = F.min()
        for t in range(T):
            E0 = 2 * self.rng.random(P) - 1
            Ep = 2 * E0 * (1 - t / T)
            for i in range(P):
                q, rr = self.rng.random(), self.rng.random()
                if abs(Ep[i]) >= 1:
                    if q >= 0.5:
                        xr = X[self.rng.integers(P)]
                        Xn = xr - self.rng.random() * np.abs(xr - 2 * self.rng.random() * X[i])
                    else:
                        Xn = (rb - X.mean(0)) - self.rng.random() * (self.rng.random())
                else:
                    J = 2 * (1 - self.rng.random())
                    if rr >= 0.5 and abs(Ep[i]) >= 0.5:
                        Xn = (rb - X[i]) - Ep[i] * np.abs(J * rb - X[i])
                    elif rr >= 0.5:
                        Xn = rb - Ep[i] * np.abs(rb - X[i])
                    else:
                        Y = rb - Ep[i] * np.abs(J * rb - X[i])
                        Xn = Y if fit(np.clip(Y, 0, 0.999)) < F[i] else Y + self.rng.normal(0, 0.1, k)
                Xn = np.clip(Xn, 0, 0.999)
                fn = fit(Xn)
                if fn < F[i]:
                    X[i], F[i] = Xn, fn
                    if fn < fb:
                        rb, fb = Xn.copy(), fn
        nh = decode(rb)
        # overwrite predecessor for CHs that have an HHO next hop
        for i, ch in enumerate(chs):
            if nh[i] >= 0:
                self.pred[ch] = nh[i]

    # ---------------- TENURE / FixedTenure
    def round_tenure(self):
        c = self.cfg; N = self.N
        if not self.cluster_init:
            self.kmedoids_init()
        changed = False
        need_route = (self.r % 20 == 0)
        for j in range(self.ncl):
            ch = self.cl_ch[j]
            mem = np.where((self.cluster == j) & self.alive)[0]
            if len(mem) == 0:
                continue
            rotate = False
            if ch < 0 or not self.alive[ch]:
                rotate = True
            else:
                if self.E[ch] <= c['e_reserve'] * c['E0']:
                    rotate = True
                elif 'trust' not in self.abl and (self.trust()[ch] < c['tau_min'] or self.blacklist[ch]):
                    rotate = True
                elif c['fixedT'] is not None:
                    rotate = (self.r - self.cl_start[j]) >= c['fixedT']
                elif 'tenure' in self.abl:
                    rotate = True
                else:
                    rotate = self.cl_spent[j] >= self.cl_theta[j]
            if rotate:
                new = self.elect_in_cluster(j, exclude=ch if (ch >= 0 and len(mem) > 1) else -1)
                if new < 0:
                    continue
                if ch >= 0 and self.r > 0:
                    self.stats['tenures'].append(self.r - self.cl_start[j])
                self.cl_ch[j] = new; self.cl_start[j] = self.r; self.cl_spent[j] = 0.0
                changed = True; need_route = True
        chs = self.cl_ch[(self.cl_ch >= 0)]
        chs = chs[self.alive[chs]]
        self.is_ch[:] = False; self.is_ch[chs] = True
        self.ch_of[:] = -1
        for j in range(self.ncl):
            ch = self.cl_ch[j]
            if ch >= 0 and self.alive[ch]:
                mem = np.where((self.cluster == j) & self.alive)[0]
                self.ch_of[mem] = ch
        mem_all = np.where((self.ch_of >= 0) & self.alive & ~self.is_ch)[0]
        bad = ~self.topo.link[mem_all, self.ch_of[mem_all]] if len(mem_all) else np.zeros(0, bool)
        self.orphans = mem_all[bad]
        self.ch_of[self.orphans] = -1
        if need_route or not hasattr(self, 'pred'):
            self.route_tree(24 + 12 * 10)
        # rekey clusters whose CH changed; ratchet others
        self._rekeyed = set()
        for j in range(self.ncl):
            ch = self.cl_ch[j]
            if ch < 0 or not self.alive[ch]:
                continue
            if self.cl_start[j] == self.r:
                mem = np.where((self.cluster == j) & self.alive)[0]; mem = mem[mem != ch]
                E0s = self.E.copy()
                kem_each_round = False
                self.rekey_cluster(ch, mem, 'tenure')
                self.stats['rotations'] += 1
                dlt = E0s - self.E
                self.rk_round += dlt
                incl = (self.cluster == j)
                Hm = float(dlt[incl].sum())
                self.h_vec[j] = np.where(incl, 0.0, dlt)
                self.cl_H[j] = Hm if not np.isfinite(self.cl_H[j]) else 0.5 * self.cl_H[j] + 0.5 * Hm
                self._rekeyed.add(j)
            elif 'ratchet' in self.abl:
                # without the hash ratchet, per-round forward secrecy needs a KEM refresh
                mem = np.where((self.cluster == j) & self.alive)[0]; mem = mem[mem != ch]
                self.rekey_cluster(ch, mem, 'tenure')
            else:
                mem = np.where((self.cluster == j) & self.alive)[0]
                self.cpu(mem, 64 * CYC['ascon_xof_B'])
        # legit re-joins (desync / reboot) feed the detector null traffic
        if self.cfg['attack'] and 'detect' not in self.abl and not ('admit' in self.abl):
            hon = np.where(self.alive & ~self.comp & ~self.is_ch)[0]
            k = self.rng.poisson(self.cfg['legit_rejoin'], size=len(hon))
            if k.any():
                nf = self.ep.update(hon, k)
                self.false_alarm += int(nf.sum())
        self.floods(chs, 'pq')
        self.data_phase(chs)
        return changed

    def post_round_tenure(self, E_before):
        used = E_before - self.E
        N = self.N
        base = used - self.rk_round
        okn = self.alive
        self.p_node[okn] = np.where(np.isfinite(self.p_node[okn]), 0.9 * self.p_node[okn] + 0.1 * base[okn], base[okn])
        self.rk_round[:] = 0.0
        for j in range(self.ncl):
            ch = self.cl_ch[j]
            if ch < 0:
                continue
            mem = np.where((self.cluster == j) & (self.alive | (E_before > 0)))[0]
            if len(mem) < 2:
                continue
            if j in getattr(self, '_rekeyed', ()):  # exclude handshake round
                continue
            u_ch = used[ch]; others = used[mem[mem != ch]]
            d = u_ch - others.mean(); P = used[mem].sum()
            self.cl_delta[j] = d if not np.isfinite(self.cl_delta[j]) else 0.8 * self.cl_delta[j] + 0.2 * d
            self.cl_P[j] = P if not np.isfinite(self.cl_P[j]) else 0.8 * self.cl_P[j] + 0.2 * P
            self.cl_spent[j] += max(d, 0.0)
            self.cl_theta[j] = self.theta_star(j)
        # BS-side revocation of flagged identities
        if 'detect' not in self.abl:
            self.blacklist |= self.ep.flag

    # ---------------- data phase (common)
    def data_phase(self, chs):
        st = self.stats
        N = self.N
        al = self.alive
        for o in getattr(self, 'orphans', np.zeros(0, int)):
            if not al[o]:
                continue
            gen = not self.comp[o]
            st['gen'] += gen
            path = self.path_from(o)
            self.cpu(o, 64 * CYC['ascon_aead_B'])
            if path is None:
                continue
            okp, lat = self.path_send(path, SZ['reading'], 'e_data')
            if okp and gen:
                st['dlv'] += 1
                st['exposed'] += int(self.legacy[o])
        for ch in chs:
            if not al[ch]:
                continue
            ms = np.where((self.ch_of == ch) & al)[0]; ms = ms[ms != ch]
            # beacon
            self.broadcast(ch, ms, SZ['beacon'], 'e_ctrl')
            ok_cnt = 0; exposed_here = 0
            if len(ms):
                gen = ~self.comp[ms]
                st['gen'] += int(gen.sum())
                okm = self.unicast_many(ms, ch, SZ['reading'], 'e_data')
                self.charge([ch], [len(ms) * GUARD_S * V * I_RX], 'e_data')
                self.cpu(ms, 64 * CYC['ascon_aead_B'])
                good = okm & gen
                ok_cnt += int(good.sum())
                exposed_here += int((good & (self.legacy[ms] | self.legacy[ch])).sum())
            own = not self.comp[ch]
            st['gen'] += own
            ok_cnt += own
            if self.legacy[ch] and own:
                exposed_here += 1
            self.cpu(ch, (len(ms) + 1) * 64 * CYC['ascon_aead_B'])
            if ok_cnt == 0:
                continue
            path = self.path_from(ch)
            if path is None:
                continue
            agg = 24 + 12 * (len(ms) + 1)
            if self.comp[ch] and self.cfg['attack'] and self.rng.random() < self.cfg['drop_p']:
                # malicious CH drops; BS observes missing aggregate
                if self._tenure_like() and 'trust' not in self.abl:
                    self.tb[ch] += 1
                continue
            okp, lat = self.path_send(path, agg, 'e_data')
            if okp:
                st['dlv'] += ok_cnt
                st['exposed'] += exposed_here
                st['lat'].append(lat)
                if self._tenure_like() and 'trust' not in self.abl:
                    self.ta[ch] += 1
            if self._tenure_like() and 'trust' not in self.abl:
                self.ta *= self.cfg['gamma'] ** (1.0 / max(len(chs), 1))
                self.tb *= self.cfg['gamma'] ** (1.0 / max(len(chs), 1))
                self.ta = np.maximum(self.ta, 1.0); self.tb = np.maximum(self.tb, 1.0)

    # ---------------- summary
    def summary(self):
        st = self.stats
        e_tot = st['e_hs'] + st['e_cpu'] + st['e_data'] + st['e_sleep'] + st['e_attack'] + st['e_ctrl']
        ins = np.where(self.comp)[0]
        det = [self.detect_round[a] for a in ins if a in self.detect_round]
        return dict(
            method=self.method, FND=self.lifetime['FND'], SL=self.lifetime['SL'],
            PDR=st['dlv'] / max(st['gen'], 1),
            J_per_reading=e_tot / max(st['dlv'], 1),
            hs_share=(st['e_hs']) / max(e_tot, 1e-12),
            e_hs=st['e_hs'], e_cpu=st['e_cpu'], e_data=st['e_data'], e_sleep=st['e_sleep'],
            e_attack=st['e_attack'], e_ctrl=st['e_ctrl'], e_total=e_tot,
            delay_ms=1e3 * float(np.mean(st['lat'])) if st['lat'] else float('nan'),
            rotations=st['rotations'], exposure=st['exposed'] / max(st['dlv'], 1),
            mean_tenure=float(np.mean(st['tenures'])) if st['tenures'] else float('nan'),
            false_alarms=self.false_alarm, n_insiders=int(self.comp.sum()),
            detected=len(det), det_delay=float(np.mean(det)) if det else float('nan'),
            alive_trace=self.alive_trace)


def run_one(method, seed, **kw):
    m = method
    kw = dict(kw)
    if method == 'FixedTenure-PQ' and kw.get('fixedT') is None:
        kw['fixedT'] = 80
    return Sim(m, seed, **kw).run()


if __name__ == '__main__':
    import time, sys
    for m in METHODS:
        t = time.time()
        s = run_one(m, 1)
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != 'alive_trace'}, round(time.time() - t, 1), 's')
