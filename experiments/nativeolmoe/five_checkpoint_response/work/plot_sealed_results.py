# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Views of completed T4 tables only. No new model/probe/threshold decisions."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = RUN / 'T4_STAGE_SEAL.json'
assert sha(seal) == (RUN / 'T4_STAGE_SEAL.sha256').read_text().split()[0]
d = json.loads(seal.read_text())
for r in d['artifacts'].values():
    assert sha(r['path']) == r['sha256']
rows = list(csv.DictReader((RUN / 'T4_TOMOGRAPHY_SUMMARY.csv').open()))
boots = list(csv.DictReader((RUN / 'T4_BOOTSTRAP_SUMMARY.csv').open()))
split = list(csv.DictReader((RUN / 'T4_SPLIT_HALF_STABILITY.csv').open()))
times = [41, 146, 503, 1740, 5033]
provenance = {'protocol': d['protocol'], 'basis': d['basis'], 'Bank_T_seal_sha256': d['Bank_T_seal_sha256'], 'source_tables': [{'path': str(RUN / n), 'sha256': sha(RUN / n)} for n in ['T4_TOMOGRAPHY_SUMMARY.csv', 'T4_BOOTSTRAP_SUMMARY.csv', 'T4_SPLIT_HALF_STABILITY.csv']], 'model_repository': d['model_repository'], 'checkpoints': d['checkpoints'], 'software_environment': d['software_environment'], 'T4_stage_seal_sha256': sha(seal), 'visualization_only': True, 'new_model_passes': 0}
folder = RUN / 'figures'
folder.mkdir(exist_ok=True)
meta = {'Software': 'matplotlib ' + matplotlib.__version__, 'Description': json.dumps(provenance)}

def subset(l, p):
    return [next((r for r in rows if int(r['tokens_B']) == t and int(r['layer']) == l and (r['probe'] == p))) for t in times]

def stable(t, l, p):
    return next((r for r in split if int(r['tokens_B']) == t and int(r['layer']) == l and (r['probe'] == p)))['status'] == 'STABLE'
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
for col, p in enumerate(['local', 'objective']):
    for l, color in [(7, '#2765A5'), (15, '#C47722')]:
        rr = subset(l, p)
        A = np.array([float(r['trace_A']) for r in rr])
        dim = np.array([float(r['d_eff']) for r in rr])
        axes[0, col].plot(times, A / A[0], '-o', color=color, label=f'L{l}')
        bs = [next((r for r in boots if int(r['tokens_B']) == t and r.get('layer') and (int(r['layer']) == l) and (r['probe'] == p) and (r['metric'] == 'd_eff'))) for t in times]
        lo = np.array([float(r['bootstrap_q025']) for r in bs])
        hi = np.array([float(r['bootstrap_q975']) for r in bs])
        axes[1, col].plot(times, dim, '-o', color=color, label=f'L{l}')
        axes[1, col].fill_between(times, lo, hi, color=color, alpha=0.13)
        for j, t in enumerate(times):
            if not stable(t, l, p):
                axes[1, col].plot(t, dim[j], 'x', color='black', markersize=7)
    axes[0, col].set(title=p.capitalize() + ' response scale', ylabel='A(t) / A(41B)', yscale='log')
    axes[1, col].set(title=p.capitalize() + ' participation dimension', ylabel='d_eff (95% paired bootstrap band)', ylim=(0.8, 6.2))
    for ax in axes[:, col]:
        ax.set_xscale('log')
        ax.set_xticks(times, [str(t) for t in times])
        ax.set_xlabel('Training tokens (B)')
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
fig.suptitle('Restricted-6D post-41B tomography; x marks unresolved split-half geometry')
fig.savefig(folder / 'P2D_SCALE_AND_EFFECTIVE_DIMENSIONS.png', dpi=180, metadata=meta)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
colors = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
for i, l in enumerate([7, 15]):
    for j, p in enumerate(['local', 'objective']):
        ax = axes[i, j]
        for t, r, c in zip(times, subset(l, p), colors):
            ax.plot(np.arange(1, 7), json.loads(r['normalized_spectrum']), '-o', color=c, label=f'{t}B')
        ax.set(title=f'L{l} {p}', xlabel='Physical response eigenvalue rank', ylabel='Eigenvalue / total trace', xticks=np.arange(1, 7), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=2)
fig.suptitle('Normalized spectra in the frozen nonorthogonal control metric')
fig.savefig(folder / 'P2D_NORMALIZED_SPECTRA.png', dpi=180, metadata=meta)
plt.close(fig)
paths = sorted(folder.glob('*.png'))
with (RUN / 'FIGURE_PROVENANCE.json').open('x') as f:
    json.dump(dict(provenance, figures=[{'path': str(p), 'sha256': sha(p)} for p in paths]), f, indent=2)
    f.write('\n')
print('SEALED-T4 FIGURES WRITTEN', len(paths))
