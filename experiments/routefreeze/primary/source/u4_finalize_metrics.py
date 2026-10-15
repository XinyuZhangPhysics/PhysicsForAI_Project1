# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen U4 bootstrap, criteria, and decision from raw arrays."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import json
import math
from pathlib import Path
import numpy as np
BASE = Path(__file__).parent
OUT = BASE / 'u4_run'
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']
raw = np.load(OUT / 'u4_scientific_arrays.npz')
T_full = raw['T_full']
T_route = raw['T_route']
B = raw['B']
T_perm = raw['T_permutation']
D_all = raw['probe']
G = raw['G']
assert T_perm.shape == (16, 4, 4) and D_all.shape == (3, 4, 4)
assert np.all(np.isfinite(T_perm)) and np.all(np.isfinite(D_all))
mask = ~np.eye(4, dtype=bool)
rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
P_full = -T_full / G[None, :]
P_route = -T_route / G[None, :]
P_B = P_full - P_route
D = D_all[1]
E_full = rms((D - P_full)[mask])
R_full = E_full / max(rms(D[mask]), 1e-12)
MAE_full = float(np.mean(np.abs(D - P_full)[mask]))
MAE_route = float(np.mean(np.abs(D - P_route)[mask]))
I_MAE = 1 - MAE_full / MAE_route
S_B = rms(P_B[mask])
rng = np.random.Generator(np.random.PCG64DXSM(202609270099))
bootstrap_indices = rng.integers(0, 16, size=(10000, 16), dtype=np.int64)
T_boot = T_perm[bootstrap_indices].mean(axis=1)
P_route_boot = -T_boot / G[None, None, :]
P_B_boot = P_full[None, :, :] - P_route_boot
route_mae_boot = np.mean(np.abs(D[None, :, :] - P_route_boot)[:, mask], axis=1)
I_boot = 1 - MAE_full / route_mae_boot
S_B_boot = np.sqrt(np.mean(np.square(P_B_boot[:, mask]), axis=1))
I_ci = np.percentile(I_boot, [2.5, 97.5]).tolist()
S_B_ci = np.percentile(S_B_boot, [2.5, 97.5]).tolist()
E_MC = rms(np.std(P_route_boot[:, mask], axis=0, ddof=1))
Q_B = S_B / max(E_full, E_MC, 1e-12)
MC_ratio = E_MC / S_B
D_RMS = rms(D[mask])
scale = (D_all.max(axis=0) - D_all.min(axis=0)) / np.maximum.reduce([np.abs(D), np.full((4, 4), D_RMS), np.full((4, 4), 1e-12)])
locality_median = float(np.median(scale[mask]))
locality_max = float(np.max(scale[mask]))
criteria = {'C1': bool(R_full <= 0.05), 'C2': bool(I_MAE >= 0.5 and I_ci[0] > 0.25), 'C3': bool(Q_B >= 3 and S_B_ci[0] > 0), 'C4': bool(MC_ratio <= 0.25), 'C5': bool(locality_median <= 0.05 and locality_max <= 0.2)}
audit = json.loads((OUT / 'u4_execution_audit.json').read_text())
assert audit['status'] == 'PASS'
if not criteria['C4']:
    decision = 'INCONCLUSIVE_MC'
elif all(criteria.values()):
    decision = 'CONFIRMED'
else:
    decision = 'NOT_CONFIRMED'
metrics = {'E_full': E_full, 'R_full': R_full, 'MAE_full': MAE_full, 'MAE_route': MAE_route, 'I_MAE': I_MAE, 'I_MAE_ci95': I_ci, 'S_B': S_B, 'S_B_ci95': S_B_ci, 'E_MC': E_MC, 'Q_B': Q_B, 'E_MC_over_S_B': MC_ratio, 'D_RMS': D_RMS, 'locality_median_delta_scale': locality_median, 'locality_max_delta_scale': locality_max, 'criteria': criteria, 'decision': decision, 'source_gradient_norms': G.tolist(), 'bootstrap_seed': 202609270099, 'bootstrap_count': 10000}
(OUT / 'u4_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
(OUT / 'u4_decision.json').write_text(json.dumps({'node': 'U4_sparse_MoE', 'decision': decision, 'criteria': criteria, 'validity_gates': 'PASS', 'metrics_file': 'u4_metrics.json'}, indent=2) + '\n')
np.savez_compressed(OUT / 'u4_bootstrap_arrays.npz', bootstrap_indices=bootstrap_indices, T_route_boot=T_boot, I_MAE_boot=I_boot, S_B_boot=S_B_boot, P_route_boot=P_route_boot)

def matrix_csv(name, matrix):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target\\source', *DOMAINS])
        for domain, row in zip(DOMAINS, matrix):
            writer.writerow([domain, *[repr(float(v)) for v in row]])
matrix_csv('u4_full_T.csv', T_full)
matrix_csv('u4_route_T.csv', T_route)
matrix_csv('u4_B.csv', B)
matrix_csv('u4_probe_primary.csv', D)
matrix_csv('u4_P_full.csv', P_full)
matrix_csv('u4_P_route.csv', P_route)
matrix_csv('u4_P_B.csv', P_B)
matrix_csv('u4_locality_delta_scale.csv', scale)
print(json.dumps(metrics, indent=2))
