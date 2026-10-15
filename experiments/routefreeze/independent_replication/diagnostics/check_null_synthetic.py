# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Synthetic common-expert-null check: identical experts => B = 0 exactly.

Mirrors the (fixed) production null logic: per-expert squared norms, not the norm of a sum.
"""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path('code').resolve()))
import numpy as np, torch
from run_u4 import SyntheticMoE
torch.manual_seed(3)
H, dm, df, n, k = (4, 6, 5, 12, 2)
syn = SyntheticMoE(H, dm, df)
base = [syn.W[0][0].detach().clone(), syn.W[0][1].detach().clone(), syn.W[0][2].detach().clone()]
for j in range(1, H):
    for t in range(3):
        syn.W[j][t].data.copy_(base[t].data)
z = torch.randn(n, dm, dtype=torch.float64)
d = torch.randn(n, dm, dtype=torch.float64)
logits = torch.randn(n, H, dtype=torch.float64)
probs = torch.softmax(logits, dim=-1)
w, idx = torch.topk(probs, k, dim=-1)
idx_np = idx.numpy()
rows = [np.nonzero(idx_np == c)[0] for c in range(H)]
slots = [np.nonzero(idx_np == c)[1] for c in range(H)]

def gvec(j, c):
    r = torch.from_numpy(rows[c])
    s = torch.from_numpy(slots[c])
    if r.numel() == 0:
        return np.zeros(3 * df * dm)
    dadj = d[r] * w[r, s].unsqueeze(-1)
    return syn.expert_grad_scaled(j, z[r], dadj).numpy().astype(np.float64)
T_full = sum((float(gvec(j, j) @ gvec(j, j)) for j in range(H)))
perms = np.array(list(itertools.permutations(range(H))))
vals = []
for pi in perms:
    inv = np.argsort(pi)
    tot = 0.0
    for j in range(H):
        g = gvec(j, int(inv[j]))
        tot += float(g @ g)
    vals.append(tot)
T_route = float(np.mean(vals))
print('T_full  =', T_full)
print('T_route =', T_route)
print('spread  =', max(vals) - min(vals))
print('ratio   =', abs(T_full - T_route) / max(abs(T_full), 1e-30))
wrong = []
for pi in perms:
    inv = np.argsort(pi)
    U = np.zeros(3 * df * dm)
    for j in range(H):
        U += gvec(j, int(inv[j]))
    wrong.append(float(U @ U))
print('previous (buggy) convention ratio =', abs(T_full - float(np.mean(wrong))) / max(abs(T_full), 1e-30))
