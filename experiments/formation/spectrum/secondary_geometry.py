# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""H3: saved-checkpoint target-MLP local Euclidean response; no objective probe."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json, csv
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from preflight import ROOT, OUT, WORK, MODEL, sha, norm, write, log

class Captured(Exception):
    pass

def run():
    decision = json.loads((OUT / 'P2H_PRIMARY_DECISION.json').read_text())
    assert decision['status'] == 'SEALED'
    for n, h in decision['primary_artifact_sha256'].items():
        assert sha(OUT / n) == h, n
    primary_sha = sha(OUT / 'P2H_PRIMARY_DECISION.json')
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    dense = model.model.layers[20].mlp
    ids = torch.from_numpy(np.load(WORK / 'p2h_banks/FE.npy'))
    hs = []

    def hook(mod, args):
        hs.append(args[0].detach())
        raise Captured()
    handle = dense.register_forward_pre_hook(hook)
    with torch.no_grad():
        for row in ids:
            try:
                model(input_ids=row[None], use_cache=False)
            except Captured:
                pass
    handle.remove()
    h = torch.cat(hs).reshape(-1, 576)
    phi = F.silu(dense.gate_proj(h)) * dense.up_proj(h)
    spec = np.load(OUT / 'P2H_CROSS_HESSIAN_SPECTRUM.npz')
    v = torch.from_numpy(spec['v'][:, :, 0])
    u = torch.from_numpy(spec['u'][:, 0])
    vphi = F.linear(phi, v)
    hu = h @ u
    c = torch.tensor([0.5, 0.5, -0.5, -0.5])
    rows = []
    grams = []
    eigvals = []
    pred = json.loads((OUT / 'P2H_PREDICTIONS.json').read_text())['predictions']
    ckpt = json.loads((OUT / 'P2H_CHECKPOINT_INDEX.json').read_text())
    assert ckpt['status'] == 'COMPLETE'
    for rec in ckpt['checkpoints']:
        if rec['epsilon'] != 0.12 or rec['update'] not in [8, 16, 32, 64]:
            continue
        assert sha(rec['path']) == rec['sha256']
        z = np.load(rec['path'])
        M = torch.from_numpy(z['M'])
        U = torch.from_numpy(z['U'])
        p = torch.softmax((h @ U)[:, None] * c, dim=-1)
        beta = (p * c).sum(-1)
        dbeta = (p * c.square()).sum(-1) - beta.square()
        rm = beta[:, None] * vphi
        ru = (dbeta * hu)[:, None] * F.linear(phi, M)
        G = torch.stack([torch.stack([rm.double().square().sum() / 8192, (rm.double() * ru.double()).sum() / 8192]), torch.stack([(rm.double() * ru.double()).sum() / 8192, ru.double().square().sum() / 8192])])
        vals, vec = torch.linalg.eigh(G)
        trace = float(vals.sum())
        deff = trace * trace / float(vals.square().sum()) if trace > 0 else 0.0
        ep = torch.tensor(pred[rec['arm']]['e_plus'], dtype=torch.float64)
        overlap = float((ep @ vec[:, -1]).square())
        rows.append(dict(seed=rec['seed'], arm=rec['arm'], epsilon=0.12, r_eta=rec['r_eta'], update=rec['update'], trace=trace, eigenvalue_1=float(vals[1]), eigenvalue_2=float(vals[0]), d_eff=deff, expert_diagonal=float(G[0, 0]), router_diagonal=float(G[1, 1]), router_expert_cross=float(G[0, 1]), projector_overlap_predicted_unstable=overlap, checkpoint_sha256=rec['sha256']))
        grams.append(G.numpy())
        eigvals.append(vals.flip(0).numpy())
    assert len(rows) == 96
    with (OUT / 'P2H_LOCAL_RESPONSE_GEOMETRY.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    np.savez(OUT / 'P2H_LOCAL_RESPONSE_GEOMETRY.npz', grams=np.stack(grams), eigenvalues=np.stack(eigvals), seed=np.array([r['seed'] for r in rows]), arm=np.array([r['arm'] for r in rows]), update=np.array([r['update'] for r in rows]), primary_decision_sha256=primary_sha)
    (OUT / 'P2H_SECONDARY_REPORT.md').write_text('# P2H secondary local response geometry\n\nComputed only after sealing H2. 96 saved checkpoints: epsilon 0.12, r_eta 0.25/1/4, eight seeds, updates 8/16/32/64.\n\nProbe: target MLP output with the Euclidean hidden-coordinate metric, averaged over all 8192 FE token positions. Columns are the unit pretraining expert direction v1 and router direction u1, ordered (M,U). No downstream loss or objective susceptibility enters this readout. G = mean_t R_t^T R_t; d_eff = trace(G)^2/trace(G^2); projector overlap is squared alignment of the leading Gram eigenvector with the frozen dynamical e_plus. The private down matrix and router use raw frozen parameter coordinates, without checkpoint-dependent rescaling.\n\nThis secondary geometry does not change the primary verdict.\n')
    write('P2H_SECONDARY_SEAL.json', dict(status='SEALED', primary_decision_sha256=primary_sha, checkpoints_evaluated=96, probe='target MLP Euclidean output response', bank='FE', objective_susceptibility=False, artifact_sha256={n: sha(OUT / n) for n in ['P2H_LOCAL_RESPONSE_GEOMETRY.csv', 'P2H_LOCAL_RESPONSE_GEOMETRY.npz', 'P2H_SECONDARY_REPORT.md']}, source_sha256=sha(__file__)))
    log('H3 COMPLETE: 96 saved-checkpoint local response Grams')
if __name__ == '__main__':
    run()
