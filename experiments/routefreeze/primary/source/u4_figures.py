# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Six frozen U4 review figures from raw result files."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE = Path(__file__).parent
OUT = BASE / 'u4_run'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
domains = ['ArXiv', 'Github', 'StackEx', 'C4']
raw = np.load(OUT / 'u4_scientific_arrays.npz')
m = json.loads((OUT / 'u4_metrics.json').read_text())
T = raw['T_full']
R = raw['T_route']
B = raw['B']
D = raw['probe']
G = raw['G']
P = -T / G[None, :]
Pr = -R / G[None, :]
mask = ~np.eye(4, dtype=bool)
labels = [f'{domains[s]}→{domains[t]}' for t in range(4) for s in range(4) if t != s]
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(D[1][mask], P[mask], label='Full', s=48, color='#136f63')
ax.scatter(D[1][mask], Pr[mask], label='Route only', s=48, color='#df6d2d', marker='x')
lo = min(D[1][mask].min(), P[mask].min(), Pr[mask].min())
hi = max(D[1][mask].max(), P[mask].max(), Pr[mask].max())
ax.plot([lo, hi], [lo, hi], ls='--', color='gray', lw=1)
ax.set(xlabel='Symmetric probe derivative', ylabel='First-order prediction', title='Off-diagonal transfer: probe vs predictions')
ax.legend()
fig.tight_layout()
fig.savefig(FIG / '01_probe_vs_predictions.png')
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
vmax = max((np.max(np.abs(x)) for x in (T, R, B)))
for ax, data, title in zip(axes, (T, R, B), ('Full T', 'Route-only T', 'Beyond-routing B')):
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4), domains, rotation=45, ha='right')
    ax.set_yticks(range(4), domains)
    ax.set(title=title, xlabel='Source', ylabel='Target')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.73, label='Gradient inner product')
fig.subplots_adjust(left=0.06, right=0.91, bottom=0.17, top=0.88, wspace=0.3)
fig.savefig(FIG / '02_transfer_heatmaps.png')
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.6))
value = m['I_MAE']
low, high = m['I_MAE_ci95']
ax.errorbar([value], [0], xerr=[[value - low], [high - value]], fmt='o', markersize=9, capsize=6, color='#136f63')
ax.axvline(0.5, ls='--', color='#af3e3e', label='Point threshold 0.50')
ax.axvline(0.25, ls=':', color='#df6d2d', label='Lower-bound threshold 0.25')
ax.set(xlabel='Full-over-route MAE improvement', yticks=[], title='16-permutation bootstrap interval', xlim=(0, 1.05))
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(FIG / '03_MAE_improvement_interval.png')
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4))
names = ['S_B', 'E_full', 'E_MC']
values = [m['S_B'], m['E_full'], m['E_MC']]
bars = ax.bar(names, values, color=['#136f63', '#527baf', '#df6d2d'])
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.4g}', ha='center', va='bottom')
ax.set(ylabel='Derivative scale', title=f"Sector resolution: Q_B={m['Q_B']:.2f}, E_MC/S_B={m['E_MC_over_S_B']:.3f}")
ax.set_ylim(0, max(values) * 1.2)
fig.tight_layout()
fig.savefig(FIG / '04_sector_resolution.png')
plt.close(fig)
rows = list(csv.DictReader((OUT / 'u4_routing_diagnostics.csv').open()))
freq = np.array([[float(next((r['selection_frequency_per_token'] for r in rows if r['split'] == 'validation' and r['domain'] == 'RedPajama' + d and (int(r['expert']) == j)))) for j in range(64)] for d in ['ArXiv', 'Github', 'StackExchange', 'C4']])
fig, ax = plt.subplots(figsize=(12, 3.8))
im = ax.imshow(freq, aspect='auto', cmap='viridis', vmin=0, vmax=max(0.5, float(freq.max())))
ax.set_yticks(range(4), domains)
ax.set_xticks(range(0, 64, 4))
ax.set(xlabel='Final-layer expert', title='Validation-domain top-8 expert selection frequency')
fig.colorbar(im, ax=ax, label='Selections per token')
fig.tight_layout()
fig.savefig(FIG / '05_expert_use_profile.png')
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
radii = np.array([5e-06, 1e-05, 2e-05])
for t in range(4):
    for s in range(4):
        if t != s:
            axes[0].plot(radii, D[:, t, s], marker='o', alpha=0.7, lw=1)
axes[0].set(xlabel='Relative radius ρ', ylabel='Probe derivative', title='12 off-diagonal pairs across radii')
den = np.maximum.reduce([np.abs(D[1]), np.full((4, 4), m['D_RMS']), np.full((4, 4), 1e-12)])
delta = (D.max(axis=0) - D.min(axis=0)) / den
axes[1].bar(range(12), delta[mask], color='#136f63')
axes[1].axhline(0.2, ls='--', color='#af3e3e', label='Maximum threshold')
axes[1].axhline(0.05, ls=':', color='#df6d2d', label='Median threshold')
axes[1].set(xlabel='Off-diagonal pair', ylabel='Δ_scale', title='Finite-difference locality')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / '06_locality.png')
plt.close(fig)
print(json.dumps({'status': 'PASS', 'figures': len(list(FIG.glob('*.png')))}))
