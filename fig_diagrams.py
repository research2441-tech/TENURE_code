"""Architecture (Fig. 1) and key-establishment sequence (Fig. 2) diagrams, grayscale."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.environ.get('TENURE_FIG', '.')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8})


def box(ax, x, y, w, h, title, lines=(), fc='0.93', lw=0.9, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec='0.1', lw=lw))
    ax.text(x + w / 2, y + h - 0.13, title, ha='center', va='top',
            fontsize=8, fontweight='bold' if bold else 'normal')
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.40 - 0.21 * i, t, ha='center', va='top', fontsize=6.8)


def arrow(ax, p, q, dashed=False, rad=0.0, lw=1.0, label=None, lpos=0.5, loff=(0, 0.08), ha='center'):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=lw, color='0.05',
                        linestyle=(0, (4, 2.5)) if dashed else '-',
                        connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]; y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha=ha, va='bottom', fontsize=6.6, style='italic',
                bbox=dict(fc='white', ec='none', pad=0.6))


def poly(ax, pts, dashed=False, lw=1.0):
    """orthogonal polyline ending in an arrow head."""
    ls = (0, (4, 2.5)) if dashed else '-'
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='0.05', lw=lw, ls=ls, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], dashed=dashed, lw=lw)


def tbox(ax, x, y, w, h, title, lines=(), fc='0.93', fs=6.1, dy=0.36):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec='0.1', lw=0.9))
    ax.text(x + w / 2, y + h - 0.16, title, ha='center', va='top', fontsize=7.1, fontweight='bold')
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - dy * i, t, ha='center', va='top', fontsize=fs)


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.16, 4.9))
    ax.set_xlim(0, 16.6); ax.set_ylim(0, 11.4); ax.axis('off')
    frame = dict(boxstyle='round,pad=0.02,rounding_size=0.12', fc='white', ec='0.4', lw=0.8, ls=(0, (2, 2)))
    ax.add_patch(FancyBboxPatch((0.2, 5.0), 16.2, 6.2, **frame))
    ax.text(0.45, 10.98, 'Base-station controller', fontsize=7.8, fontweight='bold', va='top')
    W, Y, Hh = 3.55, 7.75, 2.65
    xs = [0.5, 4.5, 8.5, 12.5]
    tbox(ax, xs[0], Y, W, Hh, 'Telemetry store',
         [r'residual energy $E_i$', 'two-sided link PRR', 'watchdog evidence', r'join counts $k_i$'])
    tbox(ax, xs[1], Y, W, Hh, 'Fragment cost model',
         [r'fragments $f(s)$', 'per-fragment ARQ', 'EFX link weight', r'rekey price $H_j$, $h_{b,j}$'])
    tbox(ax, xs[2], Y, W, Hh, 'Tenure controller',
         [r'renewal optimum $\tau_c$', r'bottleneck crossing $\tau_x$', r'max-min tenure $\tau^{\dagger}$',
          r'threshold $\theta_j$'])
    tbox(ax, xs[3], Y, W, Hh, 'CH election',
         [r'trust $T_v\geq T_{\min}$', r'PQ-capable, $E_i>E_{\mathrm{res}}$', r'energy $\times$ centrality', r'epoch tickets $t_i$'])
    for i in range(3):
        arrow(ax, (xs[i] + W, Y + 1.3), (xs[i + 1], Y + 1.3))
    tbox(ax, 1.6, 5.45, 3.6, 1.45, 'Trust and e-process',
         ['Beta reputation, flagging'], fc='0.84')
    tbox(ax, 8.5, 5.45, 3.55, 1.45, 'EFX-trust routing',
         ['shortest-path tree to BS'], fc='0.84')
    arrow(ax, (2.6, Y), (2.6, 6.9))                       # telemetry -> trust
    arrow(ax, (5.2, 6.18), (8.5, 6.18))                  # trust -> routing
    arrow(ax, (6.2, Y), (8.9, 6.9))                       # cost -> routing
    poly(ax, [(12.05, 6.18), (13.4, 6.18), (13.4, Y)])     # routing -> election/dissemination
    poly(ax, [(3.4, 5.45), (3.4, 5.2), (15.2, 5.2), (15.2, Y)])   # trust -> election
    # network tier
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 16.2, 3.75, **frame))
    ax.text(1.35, 3.75, 'IEEE 802.15.4 sensor field (Cortex-M4 nodes)', fontsize=7.8, fontweight='bold', va='top')
    tbox(ax, 0.55, 0.5, 4.1, 2.75, 'Members',
         ['ML-KEM-512 Encaps', 'Ascon-XOF ratchet each round', 'Ascon-AEAD readings',
          'admission tag per request'])
    tbox(ax, 6.25, 0.5, 4.1, 2.75, 'Cluster head',
         ['ML-KEM-512 KeyGen, Decaps', 'first-fragment tag check', 'aggregation, re-encryption',
          'ML-KEM-768 to BS key'])
    tbox(ax, 11.95, 0.5, 4.1, 2.75, 'Relays to base station',
         ['fragmented forwarding, ARQ', r'bottleneck load $p_b + h_b/\tau$', 'watchdog overhearing',
          'two-sided link reports'])
    arrow(ax, (4.65, 1.75), (6.25, 1.75)); ax.text(5.45, 1.85, 'ct, tag', ha='center', fontsize=6.4, style='italic')
    arrow(ax, (6.25, 1.05), (4.65, 1.05)); ax.text(5.45, 1.15, 'ek, tags', ha='center', fontsize=6.4, style='italic')
    arrow(ax, (10.35, 1.4), (11.95, 1.4)); ax.text(11.15, 1.5, 'aggregate', ha='center', fontsize=6.4, style='italic')
    # control (solid, down) and telemetry (dashed, up)
    poly(ax, [(xs[3] + W, Y + 1.3), (16.15, Y + 1.3), (16.15, 3.25)], lw=1.2)
    ax.text(16.05, 4.45, 'CH, $\\theta_j$, tickets,\nroutes', fontsize=6.4, ha='right', va='center')
    arrow(ax, (1.0, 3.25), (1.0, Y), dashed=True, lw=1.1)
    ax.text(1.15, 4.45, 'energy, links, joins,\nwatchdog evidence', fontsize=6.4, ha='left', va='center')
    fig.savefig(os.path.join(FIG, 'fig1_architecture.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)


def fig_seq():
    fig, ax = plt.subplots(figsize=(7.16, 4.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12.2); ax.axis('off')
    X = {'Member $i$': 1.9, 'Cluster head $c$': 7.0, 'Relays': 11.0, 'Base station': 14.2}
    for k, x in X.items():
        ax.add_patch(FancyBboxPatch((x - 1.35, 11.2), 2.7, 0.75, boxstyle='round,pad=0.02', fc='0.86', ec='0.1', lw=0.8))
        ax.text(x, 11.57, k, ha='center', va='center', fontsize=7.6, fontweight='bold')
        ax.plot([x, x], [0.3, 11.2], color='0.55', lw=0.7, ls=(0, (3, 2)), zorder=0)
    M, C, R, B = list(X.values())

    def msg(y, xa, xb, text, xl=None):
        arrow(ax, (xa, y), (xb, y), lw=1.0)
        ax.text(xl if xl else (xa + xb) / 2, y + 0.16, text, ha='center', va='bottom', fontsize=6.5,
                bbox=dict(fc='white', ec='none', pad=0.4), zorder=3)

    def note(x, y, text, ha='center'):
        ax.text(x, y, text, ha=ha, va='center', fontsize=6.5, zorder=3,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.25', lw=0.7))
    note(B, 10.45, r'elect $c$;  $t_i=\mathrm{PRF}(K_i,\mathrm{epoch})$', ha='center')
    msg(9.45, B, R, '')
    msg(9.45, R, C, '')
    ax.text((C + B) / 2, 9.62, r'tickets $\{t_i\}$, 16 B each (reliable)', ha='center', va='bottom', fontsize=6.5,
            bbox=dict(fc='white', ec='none', pad=0.4), zorder=3)
    note(C, 8.45, r'ML-KEM-768 Encaps($ek_{BS}$)')
    msg(7.45, C, R, ''); msg(7.45, R, B, '')
    ax.text((C + B) / 2, 7.62, r'$ct_{BS}$, 1088 B + tag (11 fragments)', ha='center', va='bottom', fontsize=6.5,
            bbox=dict(fc='white', ec='none', pad=0.4), zorder=3)
    note(C, 6.45, r'ML-KEM-512 KeyGen: $(ek_c, dk_c)$')
    msg(5.45, C, M, r'broadcast $ek_c$ (800 B) + tag $\sigma_i$')
    note(M + 0.3, 4.45, r'verify $\sigma_i$; Encaps($ek_c$)')
    msg(3.45, M, C, r'$ct_i$ (768 B) + tag $\mu_i$ in fragment 1')
    note(C + 0.6, 2.45, r'tag check $\rightarrow$ e-process $\rightarrow$ Decaps')
    note(11.6, 4.45, r'member and CH: $K_0=\mathrm{XOF}(ss_i\,\|\,t_i\,\|\,\mathrm{transcript})$')
    note(11.6, 3.35, r'each round: $K_{r+1}=\mathrm{XOF}(K_r)$, erase $K_r$')
    msg(1.2, M, C, r'reading under Ascon-AEAD($K_r$), 1 frame')
    fig.savefig(os.path.join(FIG, 'fig2_handshake.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_arch(); fig_seq()
