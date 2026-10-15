# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
import ast, sys, json, hashlib, time, platform, shutil, traceback, socket
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[2]
W = Path(__file__).resolve().parent
OUT = W / 'results'
OUT.mkdir(exist_ok=True)
PKG = ROOT / 'work/PhysicsForAI_Project1_Migration_v208M'
NV0 = W / 'nv0_source'
NV0OUT = Path(_p1_path('path_aa9fd874df96'))
HIST = Path(_p1_path('path_08e09cd8aadb'))
MODEL = Path(_p1_path('path_1a97f575bc3c'))
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
START = time.time()
AUDITS = []
COUNTS = dict(TRAINING_UPDATE_COUNT=0, NEW_BANK_COUNT=0, MODEL_FILES_NEWLY_DOWNLOADED=0, DATA_FILES_NEWLY_DOWNLOADED=0, BYTES_NEWLY_DOWNLOADED=0)

def deny_network(*a, **kw):
    raise RuntimeError('NETWORK_ACCESS_PROHIBITED')
socket.socket.connect = deny_network
socket.create_connection = deny_network

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def tsha(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()

def norm(t):
    return float(torch.linalg.vector_norm(t.double()))

def log(s):
    print(s, flush=True)

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def gate(ok, label, detail):
    if not ok:
        raise RuntimeError(label + ': ' + detail)
tree = ast.parse((NV0 / 'source/preflight.py').read_text())
allowed = {'PrivateMoE', 'capture', 'CrossOperator'}
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in allowed]
assert {n.name for n in nodes} == allowed
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'inherited_P2H_definitions', 'exec'))

def audit():
    expected = json.loads((NV0OUT / 'P2P_ASSET_AUDIT.json').read_text())
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=__import__('transformers').__version__)
    gate(versions == expected['versions'], 'P2P_NV1_ENVIRONMENT_AUTHORITY_MISSING', str(versions))
    for line in (NV0OUT / 'SHA256SUMS_P2P.txt').read_text().splitlines():
        h, n = line.split(maxsplit=1)
        gate(sha(NV0OUT / n.lstrip('*')) == h, 'P2P_NV1_ASSET_AUTHORITY_MISSING', n)
    index = json.loads((NV0OUT / 'P2P_SOURCE_INDEX.json').read_text())
    for rel, h in index['source_sha256'].items():
        p = NV0 / rel.replace('work/p2p/', '')
        gate(sha(p) == h, 'P2P_NV1_ASSET_AUTHORITY_MISSING', str(p))
    pre = json.loads((HIST / 'P2H_PREFLIGHT_SEAL.json').read_text())
    for n in ['P2H_DATA_BANK_SEAL.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz', 'P2H_ASSET_REUSE_AUDIT.json']:
        gate(sha(HIST / n) == pre['artifact_sha256'][n], 'P2P_NV1_ASSET_AUTHORITY_MISSING', n)
    hm = {n: h for h, n in (l.split(maxsplit=1) for l in (HIST / 'SHA256SUMS_P2H.txt').read_text().splitlines())}
    gate(sha(NV0 / 'authority/P2H_REPRODUCIBILITY_SOURCE.zip') == hm['P2H_REPRODUCIBILITY_SOURCE.zip'], 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'P2H source')
    models = []
    for rec in expected['model_files']:
        p = Path(rec['path'])
        gate(sha(p) == rec['sha256'], 'P2P_NV1_ASSET_AUTHORITY_MISSING', str(p))
        models.append(rec)
    ledger = json.loads((MODEL / 'MODEL_ASSET.json').read_text())
    gate(ledger['revision'] == MODEL.name and ledger['repo'] == 'HuggingFaceTB/SmolLM2-135M', 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'commit')
    bs = json.loads((HIST / 'P2H_DATA_BANK_SEAL.json').read_text())
    banks = {}
    records = {}
    for n, rec in bs['banks'].items():
        p = NV0 / 'inputs/banks' / f'{n}.npy'
        a = np.load(p, allow_pickle=False)
        gate(sha(p) == rec['file_sha256'] == expected['banks'][n]['sha256'] and hashlib.sha256(a.tobytes()).hexdigest() == rec['token_array_sha256'] and (list(a.shape) == rec['shape']), 'P2P_NV1_ASSET_AUTHORITY_MISSING', n)
        banks[n] = torch.from_numpy(a)
        records[n] = dict(path=str(p), sha256=sha(p), shape=list(a.shape), token_array_sha256=rec['token_array_sha256'])
    f0path = Path(expected['F0_spectrum']['path'])
    gate(sha(f0path) == expected['F0_spectrum']['sha256'], 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'F0 spectrum')
    f0 = np.load(f0path, allow_pickle=False)
    gate(f0['u'].shape == (576, 4) and f0['v'].shape == (576, 1536, 4), 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'F0 shape')
    authorities = {str(p.relative_to(PKG)): sha(p) for p in sorted(PKG.rglob('*')) if p.is_file()}
    env = dict(python=sys.executable, python_executable_sha256=sha(sys.executable), versions=versions, CPU_threads=8, interop_threads=1, deterministic_algorithms=True, autocast=False, MPS=False, dropout=False, prefix_cache=False, optimizer=False, capture_precision='float32', operator_precision='float64', network_guard='socket connection denied and HF/Transformers offline')
    write('P2P_ASSET_AUDIT.json', dict(status='EXACT_REUSED_ASSETS_VERIFIED', model_files=models, banks=records, F0_spectrum=expected['F0_spectrum'], authority_sha256=authorities, historical_seals={n: sha(HIST / n) for n in ['P2H_PREFLIGHT_SEAL.json', 'P2H_DATA_BANK_SEAL.json', 'SHA256SUMS_P2H.txt']}, historical_source_sha256=sha(NV0 / 'authority/P2H_REPRODUCIBILITY_SOURCE.zip'), environment=env, MODEL_FILES_REUSED=len(models), DATA_FILES_REUSED=10, **COUNTS))
    views = []
    for j in range(8):
        for n in [1, 17, 33, 49]:
            ids = banks[f'train_{j}'][4 * (n - 1):4 * n]
            views.append(dict(name=f'B{j}_{n}', stream=j, update=n, start=4 * (n - 1), stop=4 * n, bank=f'train_{j}', ids_sha256=tsha(ids), group='primary'))
    for q in range(8):
        ids = banks['FE'][4 * q:4 * (q + 1)]
        views.append(dict(name=f'FE_{q + 1}', start=4 * q, stop=4 * (q + 1), bank='FE', ids_sha256=tsha(ids), group='FE'))
    old = json.loads((NV0OUT / 'P2P_BATCH_VIEW_SEAL.json').read_text())['views']
    gate(all((a['ids_sha256'] == b['ids_sha256'] for a, b in zip(views, old))) and len(views) == 40, 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'historical views')
    write('P2P_BATCH_VIEW_SEAL.json', dict(status='SEALED_BEFORE_OPERATOR_OUTCOMES', views=views, sequences_per_batch=4, input_tokens=1024, valid_targets=1020, freeze_sha256=sha(PKG / '08_P2O_PROTOCOL_FREEZE/P2O_PROTOCOL_FREEZE.json'), FE_naming='FE_1..FE_8 correspond exactly to q=0..7'))
    return (banks, views, f0)

def probes(f0):
    gen = torch.Generator().manual_seed(2026100813)
    left = torch.randn(576, 4, generator=gen)
    right = torch.randn(576 * 1536, 4, generator=gen)
    q = torch.linalg.qr(torch.from_numpy(f0['v'].reshape(-1, 4)).double()).Q
    r = right.double()
    r -= q @ (q.T @ r)
    r /= torch.linalg.vector_norm(r, dim=0)
    left /= torch.linalg.vector_norm(left, dim=0)
    return (left, r.float().reshape(576, 1536, 4))

def validate(model, dense, ids, name, pr):
    log('VALIDATE ' + name)
    evidence = dict(batch=name, stage='capture', scientific_observables_computed=False)
    AUDITS.append(evidence)

    def save():
        write('P2P_CAPTURE_REPLAY_AUDIT.json', dict(status='IN_PROGRESS', audits=AUDITS, scientific_summaries_opened=False))
    h, p, g = capture(model, dense, ids)
    a = capture(model, dense, ids)
    evidence.update(replay_exact=[torch.equal(x, y) for x, y in zip((h, p, g), a)], capture_sha256=[tsha(t) for t in (h, p, g)])
    save()
    gate(all(evidence['replay_exact']), 'P2O_DETERMINISTIC_REPLAY_FAIL', name)
    del a
    moe = PrivateMoE(dense)
    errs = []
    with torch.no_grad():
        for row in ids:
            orig = model(input_ids=row[None], use_cache=False).logits
            model.model.layers[20].mlp = moe
            actual = model(input_ids=row[None], use_cache=False).logits
            errs.append(norm(actual - orig) / max(norm(orig), 1e-30))
            model.model.layers[20].mlp = dense
    evidence.update(stage='function_preservation', function_preservation_errors=errs)
    save()
    gate(max(errs) <= 1e-06, 'P2O_FUNCTION_PRESERVATION_FAIL', name)
    model.model.layers[20].mlp = moe
    d = pr[1][:, :, 0]
    u = pr[0][:, 0]
    hvp_c = torch.zeros(576)
    hvp_ct = torch.zeros_like(d)
    try:
        for row in ids:
            loss = model(input_ids=row[None], labels=row[None], use_cache=False).loss
            gm = torch.autograd.grad(loss, moe.M, create_graph=True)[0]
            hvp_c += -torch.autograd.grad((gm * d).sum(), moe.U)[0].detach() / 4
            loss = model(input_ids=row[None], labels=row[None], use_cache=False).loss
            gu = torch.autograd.grad(loss, moe.U, create_graph=True)[0]
            hvp_ct += -torch.autograd.grad((gu * u).sum(), moe.M)[0].detach() / 4
    finally:
        model.model.layers[20].mlp = dense
    op = CrossOperator(h.double(), p.double(), g.double())
    op32 = CrossOperator(h, p, g)
    ce = norm(hvp_c.double() - op.c(d.double())) / max(norm(op.c(d.double())), 1e-30)
    te = norm(hvp_ct.double() - op.ct(u.double())) / max(norm(op.ct(u.double())), 1e-30)
    evidence.update(stage='mixed_HVP', hvp_C_relative_error=ce, hvp_CT_relative_error=te, hvp_tolerance=0.0001)
    save()
    gate(max(ce, te) <= 0.0001, 'P2O_MIXED_HVP_FAIL', name)
    pairs = []
    evidence.update(stage='float64_adjoint_and_token_triangulation', probe_pairs=pairs)
    for i in range(4):
        for j in range(4):
            l = pr[0][:, i].double()
            r = pr[1][:, :, j].double()
            cv = op.c(r)
            ctu = op.ct(l)
            a = float((l * cv).sum())
            b = float((ctu * r).sum())
            token_scalar = float(-0.25 * torch.sum(op.h @ l * torch.einsum('td,df,tf->t', op.g, r, op.p)))
            nl = norm(l) * norm(cv)
            nr = norm(r) * norm(ctu)
            sadj = max(nl, nr, abs(a), abs(b), 1e-300)
            sl = max(nl, abs(a), abs(token_scalar), 1e-300)
            sr = max(nr, abs(b), abs(token_scalar), 1e-300)
            adj = abs(a - b) / sadj
            tl = abs(a - token_scalar) / sl
            tr = abs(b - token_scalar) / sr
            a32 = float((l * op32.c(pr[1][:, :, j]).double()).sum())
            b32 = float((op32.ct(pr[0][:, i]).double() * r).sum())
            pairs.append(dict(left=i, right=j, bilinear_left=a, bilinear_right=b, token_scalar=token_scalar, S_adj=sadj, S_L=sl, S_R=sr, e_adj=adj, triangulation_left=tl, triangulation_right=tr, historical_float32=dict(bilinear_left=a32, bilinear_right=b32, relative_error=abs(a32 - b32) / max(abs(a32), abs(b32), 1e-30), diagnostic_only=True)))
            save()
            gate(max(adj, tl, tr) <= 1e-09, 'P2P_NV1_DOUBLE_PRECISION_ADJOINT_FAIL', f'{name} pair {i},{j}')
    evidence.update(stage='PASS', max_e_adj=max((x['e_adj'] for x in pairs)), max_triangulation=max((max(x['triangulation_left'], x['triangulation_right']) for x in pairs)), max_historical_float32_relative=max((x['historical_float32']['relative_error'] for x in pairs)))
    save()
    torch.save(dict(h=h, phi=p, g=g), W / f'capture_{name}.pt')
    log(f"PASS {name}: adj={evidence['max_e_adj']:.4g}, triangulation={evidence['max_triangulation']:.4g}, HVP=({ce:.4g},{te:.4g})")
    return evidence

def load_model():
    m = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    m.requires_grad_(False)
    gate((m.config.hidden_size, m.config.intermediate_size, m.config.num_hidden_layers) == (576, 1536, 30), 'P2P_NV1_ASSET_AUTHORITY_MISSING', 'architecture')
    return m

def main():
    banks, views, f0 = audit()
    log('ASSET/ENVIRONMENT/VIEW SEALS PASS')
    model = load_model()
    dense = model.model.layers[20].mlp
    pr = probes(f0)
    np.savez(W / 'frozen_probes.npz', left=pr[0].numpy(), right=pr[1].numpy())
    for v in [views[0], views[32]]:
        validate(model, dense, banks[v['bank']][v['start']:v['stop']], v['name'], pr)
    write('P2P_CAPTURE_REPLAY_AUDIT.json', dict(status='PASS', audits=AUDITS, scientific_summaries_opened=False))
    old = json.loads((NV0OUT / 'P2P_CAPTURE_REPLAY_AUDIT.json').read_text())['audits'][0]['transpose_audit']
    replay = [dict(left=a['left'], right=a['right'], historical_relative=a['relative_error'], current_relative=b['historical_float32']['relative_error'], exact_float32_scalar_replay=a['bilinear_left'] == b['historical_float32']['bilinear_left'] and a['bilinear_right'] == b['historical_float32']['bilinear_right']) for a, b in zip(old, AUDITS[0]['probe_pairs'])]
    write('P2P_NV0_DIAGNOSTIC_COMPARISON.json', dict(diagnostic_only=True, pairs=replay, all_exact=all((x['exact_float32_scalar_replay'] for x in replay))))
    write('P2P_NV1_NUMERICAL_VALIDATION_PASS.json', dict(status='P2P_NV1_PASS', validation_only=True, validated_batches=['B0_1', 'FE_1'], probe_seed=2026100813, left_probe_count=4, right_probe_count=4, adjoint_normwise_tolerance=1e-09, token_scalar_triangulation_tolerance=1e-09, inherited_HVP_tolerance=0.0001, audit_sha256=sha(OUT / 'P2P_CAPTURE_REPLAY_AUDIT.json'), asset_audit_sha256=sha(OUT / 'P2P_ASSET_AUDIT.json'), source_sha256=sha(__file__), probe_sha256=sha(W / 'frozen_probes.npz'), elapsed_seconds=time.time() - START, **COUNTS))
    (OUT / 'P2P_NV1_PASS.sha256').write_text(sha(OUT / 'P2P_NV1_NUMERICAL_VALIDATION_PASS.json') + '  P2P_NV1_NUMERICAL_VALIDATION_PASS.json\n')
    log('NV1 PASS ARTIFACT SEALED')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        label = str(e).split(':')[0] if str(e).startswith(('P2O_', 'P2P_NV1_')) else 'P2P_NV1_REQUIRED_ARTIFACT_OR_EXECUTION_FAIL'
        write('P2P_NV1_NONSCIENTIFIC_STOP.json', dict(status=label, scientific_verdict=None, error=str(e), type=type(e).__name__, reached_batches=[a['batch'] for a in AUDITS], elapsed_seconds=time.time() - START, **COUNTS))
        sys.exit(1)
