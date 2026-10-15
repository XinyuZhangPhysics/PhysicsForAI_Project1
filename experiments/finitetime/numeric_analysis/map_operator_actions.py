# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Exact matrix-free maps from recovered frozen h/phi/g factors.

Use full M shape (576,1536) and U shape (576,). Captures are float32
archival arrays promoted to float64 before any operator arithmetic.
"""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import numpy as np

def load_factors(directory, stream):
    caps = []
    for step in range(1, 33):
        with np.load(Path(directory) / f'stream{stream}_step{step:02}.npz', allow_pickle=False) as z:
            caps.append(tuple((z[k].astype(np.float64) for k in ['h', 'phi', 'g'])))
    return caps

def action_c(cap, m):
    h, phi, g = cap
    return -0.25 * ((g * (phi @ m.T)).sum(axis=1) @ h)

def action_ct(cap, u):
    h, phi, g = cap
    return -0.25 * (g.T @ ((h @ u)[:, None] * phi))

def ordered_product(caps, m0, u0, arm):
    m = np.asarray(m0, dtype=np.float64).copy()
    u = np.asarray(u0, dtype=np.float64).copy()
    eta_m, eta_u = (arm['eta_M'], arm['eta_U'])
    a_m = 1 - eta_m * arm['mu']
    a_u = 1 - eta_u * arm['lambda_']
    for cap in caps:
        cm = action_c(cap, m)
        ct = action_ct(cap, u)
        m, u = (a_m * m + eta_m * ct, a_u * u + eta_u * cm)
    return (m, u)

def same_sequence_mean_product(caps, m0, u0, arm):
    assert len(caps) == 32
    m = np.asarray(m0, dtype=np.float64).copy()
    u = np.asarray(u0, dtype=np.float64).copy()
    eta_m, eta_u = (arm['eta_M'], arm['eta_U'])
    a_m = 1 - eta_m * arm['mu']
    a_u = 1 - eta_u * arm['lambda_']
    for _ in range(32):
        cm = sum((action_c(cap, m) for cap in caps)) / 32
        ct = sum((action_ct(cap, u) for cap in caps)) / 32
        m, u = (a_m * m + eta_m * ct, a_u * u + eta_u * cm)
    return (m, u)
