# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
import ast, csv, hashlib, json, platform, socket, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM
BASE = Path(_p1_path('path_65e87eb57756'))
ROOT = Path(_p1_path('path_fd01557d84f7'))
OUT = ROOT / 'outputs/v285M_archive_completion'
STAGE = BASE / 'Project1_B2_B3_SelectedArtifacts'
PRIOR = BASE / 'Project1_B2_B3_Reanalysis'
P = BASE / '2026-10-09/g-ju/outputs/P2P_NV1_FRESH_P2P_COMPLETE'
Q = BASE / '2026-10-09/an/outputs/P2Q_RANDOM_COCYCLE_COMPLETE'
H = BASE / '2026-10-08/jie/outputs'
EPS = 1e-30
TIMES = [1, 2, 4, 8, 16, 32]
HASHES = {}
AUDIT = {}
CAPTURE_AUDIT = []
START = time.monotonic()
(OUT / 'work').mkdir(exist_ok=True)

def log(s):
    print(s, flush=True)

def extract(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())
source = BASE / 'Project1_B2_FinalClosure/analysis_b2_final_closure.py'
extract(source, {'sha', 'verify', 'tsha', 'norm', 'relative', 'near', 'verify_inputs', 'ct_many', 'c_many', 'checkpoint', 'metrics'})
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def deny(*a, **k):
    raise RuntimeError('NETWORK_ACCESS_PROHIBITED')
socket.socket.connect = deny
socket.create_connection = deny
verify_inputs()
HASHES[str(source)] = sha(source)
(OUT / 'FINITE_TIME_SOURCE_HASHES.json').write_text(json.dumps(HASHES, indent=2))
preds = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
cp = {(r['seed'], r['arm'], r['update']): r for r in json.loads((H / 'P2H_CHECKPOINT_INDEX.json').read_text())['checkpoints'] if r['update'] in [0] + TIMES}
seeds = torch.stack(torch.load(H.parent / 'work/p2h_private_seeds.pt', weights_only=True)).double()
banks = {f'train_{j}': torch.from_numpy(np.load(P / f'inputs/banks/train_{j}.npy', allow_pickle=False)) for j in range(8)}
oldcap = {(r['stream'], r['update']): r for r in json.loads((Q / 'P2Q_OPERATOR_ACTION_AUDIT.json').read_text())['batches']}
views = json.loads((Q / 'P2Q_BATCH_SEQUENCE_SEAL.json').read_text())['views']
for v in views:
    assert tsha(banks[f"train_{v['stream']}"][v['start']:v['stop']]) == v['ids_sha256']
with np.load(P / 'P2P_POOLED_SUBSPACE.npz', allow_pickle=False) as z:
    qm = torch.from_numpy(z['Q_M'].copy()).double()
    qu = torch.from_numpy(z['Q_U'].copy()).double()
etaM = torch.tensor([p['eta_M'] for p in preds], dtype=torch.float64)
etaU = torch.tensor([p['eta_U'] for p in preds], dtype=torch.float64)
aM = 1 - etaM * torch.tensor([p['mu'] for p in preds], dtype=torch.float64)
aU = 1 - etaU * torch.tensor([p['lambda_'] for p in preds], dtype=torch.float64)
ref = pq.read_table(BASE / 'Project1_B2_FinalClosure/B2_M1_PER_CASE_ERRORS.parquet').to_pylist()
prior = pq.read_table(PRIOR / 'B2_PER_CASE_ERRORS.parquet').to_pylist()
ref = {(r['seed'], r['arm']): r for r in ref if r['update'] == 32}
prior = {(r['seed'], r['arm']): r for r in prior if r['update'] == 32 and r['model'] == 'M3_FULL_TIME_ORDERED'}
model = AutoModelForCausalLM.from_pretrained(str(P / 'inputs/model'), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
model.requires_grad_(False)
dense = model.model.layers[20].mlp
rows = []
checks = []
for j in range(8):
    caps = []
    for n in range(1, 33):
        hp = capture(model, dense, banks[f'train_{j}'][4 * (n - 1):4 * n])
        hs = {name: tsha(t) for name, t in zip(['h', 'phi', 'g'], hp)}
        assert hs == oldcap[j, n]['capture_sha256'], (j, n)
        CAPTURE_AUDIT.append(dict(stream=j, update=n, hashes=hs, match=True))
        np.savez_compressed(ROOT / f'work/recovered_maps/stream{j}_step{n:02}.npz', **{name: t.numpy() for name, t in zip(['h', 'phi', 'g'], hp)})
        caps.append(tuple((t.double() for t in hp)))
    log(f'stream {j}: all 32 capture hashes matched')
    initial_m, initial_u = checkpoint(cp, j, 0, 0)
    assert torch.equal(initial_m, seeds[j]) and torch.count_nonzero(initial_u) == 0
    ms = seeds[j][None].repeat(15, 1, 1)
    us = torch.zeros(15, 576, dtype=torch.float64)
    for cap in caps:
        cm = c_many(cap, ms)
        ct = ct_many(cap, us)
        ms, us = (aM[:, None, None] * ms + etaM[:, None, None] * ct, aU[:, None] * us + etaU[:, None] * cm)
    log(f'stream {j}: ordered full-space propagation complete')
    Gmean = torch.zeros(576, 576, dtype=torch.float64)
    for i, (hi, pi, gi) in enumerate(caps):
        for k in range(i, 32):
            hk, pk, gk = caps[k]
            token = gi @ gk.T * (pi @ pk.T)
            block = hi.T @ token @ hk / 16 / 1024
            Gmean += block if i == k else block + block.T
    cs = sum((CrossOperator(*cap).c(seeds[j]) for cap in caps)) / 32
    r = torch.zeros(15, 576, dtype=torch.float64)
    u = r.clone()
    for n in range(32):
        r, u = (aM[:, None] * r + u, aU[:, None] * u + etaU[:, None] * (aM[:, None] ** n * cs[None] + etaM[:, None] * (r @ Gmean.T)))
    meanm = aM[:, None, None] ** 32 * seeds[j][None]
    for cap in caps:
        meanm += etaM[:, None, None] * ct_many(cap, r) / 32
    for k in range(15):
        om, ou = checkpoint(cp, j, k, 32)
        seqmetrics = metrics(ms[k], us[k], om, ou, seeds[j], qm, qu)
        meanmetrics = metrics(meanm[k], u[k], om, ou, seeds[j], qm, qu)
        for field in ['E_state_full', 'E_M', 'E_U']:
            near(seqmetrics[field], prior[j, k][field], f'sealed sequence {j}/{k}/{field}')
        den = {'E_state_full': float(torch.sqrt((om - seeds[j]).square().sum() + ou.square().sum())), 'E_M': norm(om - seeds[j]), 'E_U': norm(ou)}
        for field, sector in [('E_state_full', 'full_state'), ('E_M', 'expert_private'), ('E_U', 'router_private')]:
            rf = ref[j, k][field]
            sq = seqmetrics[field]
            sm = meanmetrics[field]
            rows.append(dict(stream=j, arm=k, population='primary' if j < 6 else 'replication', update=32, epsilon=preds[k]['epsilon'], r_eta=preds[k]['r_eta'], sector=sector, original_metric=field, target_increment_norm=den[field], sequence_absolute_error=sq * den[field], seqmean_absolute_error=sm * den[field], refmean_absolute_error=rf * den[field], sequence_original_error=sq, seqmean_original_error=sm, refmean_original_error=rf, refmean_over_sequence=rf / max(sq, EPS), seqmean_over_sequence=sm / max(sq, EPS), refmean_over_seqmean=rf / max(sm, EPS)))
    log(f'stream {j}: same-sequence mean and 45 metric rows verified')
    with (OUT / 'FINITE_TIME_THREE_WAY_CASES.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / 'FINITE_TIME_CAPTURE_HASH_VERIFICATION.json').write_text(json.dumps(CAPTURE_AUDIT, indent=2))
    del caps, ms, us, meanm, Gmean
(OUT / 'FINITE_TIME_RECOVERY_VERIFICATION.json').write_text(json.dumps(dict(status='PASS', recovered_maps=256, cases=120, rows=len(rows), full_expert_coordinates=576 * 1536, router_coordinates=576, sequence_error_matches_sealed_per_case=True, elapsed_seconds=time.monotonic() - START, audit=AUDIT), indent=2))
log('FINITE TIME COMPLETE')
