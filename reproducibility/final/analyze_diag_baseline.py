from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / 'evidence/generic_local_dynamics/generic_local_dynamics_long.csv'
OUTDIR = ROOT / 'analysis/astra_r2_node1'
OUTDIR.mkdir(parents=True, exist_ok=True)

MU = 0.017010007119361258
ETA_X = 11.757784614467004
ETA_U = 11.757784614467004
KEYS = ['side', 'epsilon', 'direction', 'seed', 'amplitude', 'A']

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

df = pd.read_csv(IN)
init = df[df.step == 0].set_index(KEYS)
rows = []
for _, r in df[df.step > 0].iterrows():
    key = tuple(r[k] for k in KEYS)
    r0 = init.loc[key]
    n = int(r.step)
    a_x = 1 - ETA_X * MU
    a_u = 1 - ETA_U * float(r.lambda_)
    obs, full, diag, z0 = [], [], [], []
    for j in range(1, 5):
        m0, u0 = float(r0[f'm{j}_obs']), float(r0[f'u{j}_obs'])
        mo, uo = float(r[f'm{j}_obs']), float(r[f'u{j}_obs'])
        mf, uf = float(r[f'm{j}_lin']), float(r[f'u{j}_lin'])
        md, ud = a_x**n * m0, a_u**n * u0
        z0 += [m0, u0]
        obs += [mo, uo]
        full += [mf, uf]
        diag += [md, ud]
    z0 = np.asarray(z0)
    obs = np.asarray(obs)
    full = np.asarray(full)
    diag = np.asarray(diag)
    inc_obs = obs - z0
    inc_full = full - z0
    inc_diag = diag - z0
    den_obs = np.linalg.norm(inc_obs)
    den_full = np.linalg.norm(inc_full)
    rows.append({
        'side': r.side,
        'epsilon': r.epsilon,
        'direction': r.direction,
        'seed': r.seed,
        'amplitude': r.amplitude,
        'A': r.A,
        'step': n,
        'a_X': a_x,
        'a_U': a_u,
        'e4_full_obsden': np.linalg.norm(obs - full) / den_obs,
        'e4_diag_obsden': np.linalg.norm(obs - diag) / den_obs,
        'e4_full_fullref': np.linalg.norm(obs - full) / den_full,
        'e4_diag_fullref': np.linalg.norm(obs - diag) / den_full,
        'cos4_full': cosine(inc_obs, inc_full),
        'cos4_diag': cosine(inc_obs, inc_diag),
        'norm_inc_obs': den_obs,
        'norm_inc_full': den_full,
        'norm_inc_diag': np.linalg.norm(inc_diag),
    })

out = pd.DataFrame(rows)
out.to_csv(OUTDIR / 'generic_local_dynamics_diag_baseline.csv', index=False)

def stats(x):
    x = np.asarray(x, dtype=float)
    return {
        'median': float(np.median(x)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'q05': float(np.quantile(x, 0.05)),
        'q95': float(np.quantile(x, 0.95)),
    }

summary = {
    'definition': {
        'diag_map': 'diag((1-eta_X*mu) I_X, (1-eta_U*lambda) I_U)',
        'eta_X': ETA_X,
        'eta_U': ETA_U,
        'mu': MU,
        'a_X': 1 - ETA_X * MU,
        'scope': 'leading-four singular-pair projected increments from retained saved coefficients; no full-state-vector reconstruction',
        'e_model_obsden': '||P4[(z_n-z0)-(z_model,n-z0)]||_2 / ||P4(z_n-z0)||_2',
    },
    'all_nonzero': {},
    'by_side': {},
    'endpoint': {},
}
for col in ['e4_full_obsden', 'e4_diag_obsden', 'e4_full_fullref', 'e4_diag_fullref', 'cos4_full', 'cos4_diag']:
    summary['all_nonzero'][col] = stats(out[col])
for side, g in out.groupby('side'):
    summary['by_side'][side] = {col: stats(g[col]) for col in ['e4_full_obsden', 'e4_diag_obsden', 'cos4_full', 'cos4_diag']}
for side, g in out[out.step == 28].groupby('side'):
    summary['endpoint'][side] = {col: stats(g[col]) for col in ['e4_full_obsden', 'e4_diag_obsden', 'cos4_full', 'cos4_diag']}

(OUTDIR / 'generic_local_dynamics_diag_baseline_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
