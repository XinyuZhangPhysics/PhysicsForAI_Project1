# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Figures for the U4 replication report (mandatory figure list, preregistration Sec. 24).

Lives in diagnostics/ on purpose: code/ is the sealed source tree.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE / 'code'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from u4_common import PATHS
RAW = PATHS['raw']
FIG = PATHS['figures']
FIG.mkdir(parents=True, exist_ok=True)
DOM = ['ArXiv', 'Github', 'StackEx', 'C4']

def main():
    raw = json.loads((RAW / 'u4_run_raw.json').read_text())
    met = json.loads((RAW / 'u4_metrics.json').read_text())
    T_full = np.array(met['T_full_4x4'])
    T_route = np.array(met['T_route_4x4'])
    B = np.array(met['B_4x4'])
    D = np.array(met['D_probe_primary'])
    Pf = np.array(met['P_full'])
    Pr = np.array(met['P_route'])
    boot = np.load(RAW / 'u4_bootstrap_arrays.npz')
    n = T_full.shape[0]
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    d_o = np.array([D[i, j] for i, j in off])
    pf_o = np.array([Pf[i, j] for i, j in off])
    pr_o = np.array([Pr[i, j] for i, j in off])
    ax.scatter(d_o, pf_o, s=42, c='#1f77b4', label='$P^{\\rm full}$', zorder=3)
    ax.scatter(d_o, pr_o, s=42, c='#d62728', marker='s', label='$P^{\\rm route}$', zorder=3)
    lo = float(min(d_o.min(), pf_o.min(), pr_o.min()))
    hi = float(max(d_o.max(), pf_o.max(), pr_o.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, label='identity')
    ax.set_xlabel('$D^{\\rm probe}$  ($\\rho=10^{-5}$)')
    ax.set_ylabel('prediction')
    ax.set_title('Off-diagonal pairs: probe vs first-order predictions')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_probe_vs_predictions.png', dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, M, name in zip(axs, (T_full, T_route, B), ('$T^{\\rm full}$', '$\\widehat T^{\\rm route}$', '$\\widehat B$')):
        im = ax.imshow(M, cmap='RdBu_r', origin='upper')
        ax.set_xticks(range(n))
        ax.set_xticklabels(DOM, rotation=30, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(DOM)
        ax.set_title(name + '\n(rows: target, cols: source)')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_heatmaps.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(boot['I_MAE'], bins=60, color='#2ca02c', alpha=0.85)
    lo_ci, hi_ci = met['I_MAE_CI95']
    ax.axvline(met['I_MAE'], color='k', lw=2, label=f"point {met['I_MAE']:.4f}")
    ax.axvline(lo_ci, color='k', ls=':', lw=1.5, label=f'95% CI [{lo_ci:.4f}, {hi_ci:.4f}]')
    ax.axvline(hi_ci, color='k', ls=':', lw=1.5)
    ax.set_xlabel('$I_{\\rm MAE}$ bootstrap replicate')
    ax.set_ylabel('count')
    ax.set_title('Permutation bootstrap of the full-over-routing advantage')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_I_MAE_bootstrap.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    vals = [met['S_B'], met['E_full'], met['E_MC']]
    ax.bar(['$S_B$', '$E_{\\rm full}$', '$E_{\\rm MC}$'], vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.3e}', ha='center', va='bottom', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('private-sector size vs error scales  ($Q_B$=%.3f, $E_{\\rm MC}/S_B$=%.3f)' % (met['Q_B'], met['E_MC_over_S_B']))
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_scales.png', dpi=150)
    plt.close(fig)
    diag = json.loads((RAW / 'u4_run_raw.json').read_text())['routing_diagnostics']
    keys = [k for k in diag if '|' in k and (not k.startswith('overlap'))]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.2))
    freq = np.array([diag[k]['selection_freq_max'] for k in keys])
    ent = np.array([diag[k]['top8_route_entropy'] for k in keys])
    lab = [k.replace('|', '/').replace('RedPajama', '') for k in keys]
    axs[0].bar(range(len(lab)), freq, color='#4c72b0')
    axs[0].set_xticks(range(len(lab)))
    axs[0].set_xticklabels(lab, rotation=40, ha='right')
    axs[0].set_title('max expert selection frequency')
    axs[1].bar(range(len(lab)), ent, color='#55a868')
    axs[1].set_xticks(range(len(lab)))
    axs[1].set_xticklabels(lab, rotation=40, ha='right')
    axs[1].set_title('mean top-8 route entropy')
    for a in axs:
        a.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_routing_profile.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    radii = raw['stages']['I_probe']['D_probe']
    rs = sorted(radii.keys(), key=float)
    x = np.arange(len(off))
    for r in rs:
        ax.plot(x, [radii[r][i][j] for i, j in off], 'o-', ms=4, label='$\\rho$=' + r)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{DOM[i]}←{DOM[j]}' for i, j in off], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('$D^{\\rm probe}$')
    ax.set_title('probe-scale locality (12 off-diagonal pairs)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_locality.png', dpi=150)
    plt.close(fig)
    print('figures written to', FIG)
if __name__ == '__main__':
    main()
