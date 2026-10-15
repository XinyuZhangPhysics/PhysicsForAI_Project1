# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""WP1 independent numerical pilot. Run only after build_banks.py succeeds."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import transformers
import huggingface_hub
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM
from validated_preflight_v1p1 import freeze_all, get_target, local_vjp_one, mem_snapshot, sync
ROOT = Path(__file__).resolve().parent
MODEL = 'allenai/OLMoE-1B-7B-0924'
SHA = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
SEED = 20260928
NESTED = (4, 8, 16, 32)
GRID = tuple((2.0 ** k for k in range(-18, -7)))

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_sha(tensor):
    h = hashlib.sha256()
    a = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    h.update(a.tobytes())
    return h.hexdigest()

def log(event, **kwargs):
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **kwargs}
    with open(ROOT / 'WP1_Pilot_Run.log', 'a') as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + '\n')
    print(event, kwargs, flush=True)

def flatten_dot(a, b):
    return float(np.dot(a[0].ravel(), b[0].ravel()) + np.dot(a[1].ravel(), b[1].ravel()))

def grad_norm(a):
    return math.sqrt(max(flatten_dot(a, a), 0.0))

def expert_dots(a, b):
    return np.einsum('ijk,ijk->i', a[0], b[0], optimize=True) + np.einsum('ijk,ijk->i', a[1], b[1], optimize=True)

def probe_one(model, mlp, experts, ids, device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    captured = {}

    def hook(module, args, output):
        captured['h'] = args[0].detach()
        leaf = output.detach().requires_grad_(True)
        captured['leaf'] = leaf
        return leaf
    handle = mlp.register_forward_hook(hook)
    try:
        output = model(input_ids=x, labels=x, use_cache=False)
        output.loss.backward()
    finally:
        handle.remove()
    h = captured['h']
    r = captured['leaf'].grad.detach()
    with torch.no_grad():
        _, rw, ri = mlp.gate(h.reshape(-1, h.shape[-1]))
    grads = {}
    loads = np.bincount(ri.detach().cpu().numpy().ravel(), minlength=64)
    for i in np.nonzero(loads)[0]:
        gu, dn, _ = local_vjp_one(experts, int(i), h, r, rw, ri)
        grads[int(i)] = (gu.float().cpu().numpy().copy(), dn.float().cpu().numpy().copy())
    sync(device)
    loss = float(output.loss.detach().float().cpu())
    del x, output, captured, h, r, rw, ri
    return (grads, loads, loss)

def empty_grad(experts):
    return (np.zeros(tuple(experts.gate_up_proj.shape), dtype=np.float32), np.zeros(tuple(experts.down_proj.shape), dtype=np.float32))

def accumulate(acc, item):
    for i, (gu, dn) in item.items():
        acc[0][i] += gu
        acc[1][i] += dn

def gram_matrix(items):
    n = len(items)
    gram = np.zeros((n, n), dtype=np.float64)
    for i, x in enumerate(items):
        for j in range(i + 1):
            y = items[j]
            value = 0.0
            for k in x.keys() & y.keys():
                value += float(np.dot(x[k][0].ravel(), y[k][0].ravel()))
                value += float(np.dot(x[k][1].ravel(), y[k][1].ravel()))
            gram[i, j] = gram[j, i] = value
    return gram

def bootstrap_rse(gram, n, reps=1000):
    rng = np.random.default_rng(SEED + n)
    counts = rng.multinomial(n, [1 / n] * n, size=reps).astype(np.float64)
    norms = np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', counts, gram[:n, :n], counts, optimize=True), 0)) / n
    return float(np.std(norms, ddof=1) / max(np.mean(norms), 1e-30))

def collect_bank(model, mlp, experts, bank, domain, label, device, bootstrap=False):
    t0 = time.perf_counter()
    accum = empty_grad(experts)
    snapshots = {}
    items = [] if bootstrap else None
    loads_total = np.zeros(64, dtype=np.int64)
    losses = []
    for j, row in enumerate(bank, 1):
        start = time.perf_counter()
        grads, loads, loss = probe_one(model, mlp, experts, row['token_ids'], device)
        accumulate(accum, grads)
        if bootstrap:
            items.append(grads)
        loads_total += loads
        losses.append(loss)
        if bootstrap and j in NESTED:
            snapshots[j] = (accum[0].copy() / j, accum[1].copy() / j)
        log('sequence_gradient', domain=domain, bank=label, index=j, seconds=time.perf_counter() - start, sequence_count=1, token_count=256, active_experts=int(np.count_nonzero(loads)), assignments=int(loads.sum()), memory=mem_snapshot(device))
    mean = (accum[0] / len(bank), accum[1] / len(bank))
    del accum
    study = None
    if bootstrap:
        gram = gram_matrix(items)
        reference = snapshots[32]
        refnorm = grad_norm(reference)
        study = []
        for n in NESTED:
            current = snapshots[n]
            diff = (current[0] - reference[0], current[1] - reference[1])
            by_expert_ref = expert_dots(reference, reference)
            by_expert_diff = expert_dots(diff, diff)
            study.append({'n': n, 'whole_relative_discrepancy': grad_norm(diff) / max(refnorm, 1e-30), 'expert_relative_discrepancy': (np.sqrt(np.maximum(by_expert_diff, 0)) / np.maximum(np.sqrt(np.maximum(by_expert_ref, 0)), 1e-30)).tolist(), 'bootstrap_relative_se_gradient_norm': bootstrap_rse(gram, n), 'routing_coverage': int(np.count_nonzero(loads_total[:]))})
        del items, gram, snapshots
    path = ROOT / f'gradient_{domain}_{label}32.pt'
    torch.save({'gate_up': torch.from_numpy(mean[0]), 'down': torch.from_numpy(mean[1]), 'domain': domain, 'bank': label, 'count': len(bank)}, path)
    info = {'artifact': path.name, 'sha256': file_sha(path), 'norm': grad_norm(mean), 'mean_loss': statistics.mean(losses), 'sequence_count': len(bank), 'token_count': len(bank) * 256, 'active_experts': int(np.count_nonzero(loads_total)), 'assignments': int(loads_total.sum()), 'routing_assignments_by_expert': loads_total.tolist(), 'seconds': time.perf_counter() - t0, 'memory': mem_snapshot(device)}
    if study is not None:
        info['nested_study'] = study
    log('bank_gradient_complete', domain=domain, bank=label, seconds=info['seconds'], sequence_count=len(bank), token_count=len(bank) * 256, active_experts=info['active_experts'], assignments=info['assignments'], memory=info['memory'])
    del mean
    gc.collect()
    return info

def load_grad(domain, label):
    data = torch.load(ROOT / f'gradient_{domain}_{label}32.pt', map_location='cpu', weights_only=False)
    return (data['gate_up'].numpy(), data['down'].numpy())

def eval_bank(model, mlp, bank, device, routes=False):
    losses = []
    route_hash = hashlib.sha256()
    for row in bank:
        x = torch.tensor([row['token_ids']], dtype=torch.long, device=device)
        stash = {}
        handle = None
        if routes:

            def prehook(module, args):
                stash['h'] = args[0].detach()
            handle = mlp.register_forward_pre_hook(prehook)
        try:
            with torch.no_grad():
                output = model(input_ids=x, labels=x, use_cache=False)
                if routes:
                    _, _, ri = mlp.gate(stash['h'].reshape(-1, stash['h'].shape[-1]))
                    route_hash.update(ri.cpu().numpy().tobytes())
            losses.append(float(output.loss.float().cpu()))
        finally:
            if handle is not None:
                handle.remove()
    sync(device)
    return (statistics.mean(losses), route_hash.hexdigest() if routes else None)

def repeated_floor(model, mlp, bank, domain, device):
    vals, routes = ([], [])
    t0 = time.perf_counter()
    for j in range(10):
        v, r = eval_bank(model, mlp, bank, device, routes=True)
        vals.append(v)
        routes.append(r)
        log('no_update_repeat', domain=domain, repetition=j + 1, loss=v, route_sha256=r)
    floor = max(1e-12, 10 * max((abs(v - statistics.median(vals)) for v in vals)))
    out = {'losses': vals, 'route_sha256': routes, 'routes_exactly_equal': len(set(routes)) == 1, 'loss_floor': floor, 'exact_repeats_not_certified_fp_bound': len(set(vals)) == 1, 'seconds': time.perf_counter() - t0}
    return out

def step_study(model, mlp, experts, bank, domain, source_info, base, base_hashes, floor, device):
    source = load_grad(domain, 'S')
    ggu = torch.from_numpy(source[0]).to(device)
    gdn = torch.from_numpy(source[1]).to(device)
    gnorm2 = float((ggu.square().sum() + gdn.square().sum()).cpu())
    theta_norm2 = float((base[0].float().square().sum() + base[1].float().square().sum()).cpu())
    base_loss = statistics.median(floor['losses'])
    rows = []
    for intended in GRID:
        t0 = time.perf_counter()
        eta = intended * math.sqrt(theta_norm2 / gnorm2)
        with torch.no_grad():
            new_gu = (base[0].float() - eta * ggu).to(torch.bfloat16)
            new_dn = (base[1].float() - eta * gdn).to(torch.bfloat16)
            dgu = new_gu.float() - base[0].float()
            ddn = new_dn.float() - base[1].float()
            experts.gate_up_proj.copy_(new_gu)
            experts.down_proj.copy_(new_dn)
            delta_norm2 = float((dgu.square().sum() + ddn.square().sum()).cpu())
            residual2 = float(((dgu + eta * ggu).square().sum() + (ddn + eta * gdn).square().sum()).cpu())
            actual_prediction = float((dgu * ggu).sum().cpu() + (ddn * gdn).sum().cpu())
            unchanged = float(((new_gu == base[0]).sum() + (new_dn == base[1]).sum()).cpu()) / (new_gu.numel() + new_dn.numel())
        try:
            after, _ = eval_bank(model, mlp, bank, device)
        finally:
            with torch.no_grad():
                experts.gate_up_proj.copy_(base[0])
                experts.down_proj.copy_(base[1])
        measured = after - base_loss
        ideal_prediction = -eta * gnorm2
        quant = math.sqrt(residual2) / max(eta * math.sqrt(gnorm2), 1e-30)
        cosine = -actual_prediction / max(math.sqrt(delta_norm2 * gnorm2), 1e-30)
        error = abs(measured - actual_prediction) / max(abs(actual_prediction), floor['loss_floor'])
        usable = abs(measured) > floor['loss_floor'] and measured < 0 and (error <= 0.1)
        ideal = quant <= 0.05 and cosine >= 0.999
        row = {'intended_rho': intended, 'eta': eta, 'actual_rho': math.sqrt(delta_norm2 / theta_norm2), 'quantization_distortion': quant, 'cosine_alignment': cosine, 'unchanged_parameter_fraction': unchanged, 'measured_loss_change': measured, 'ideal_first_order_prediction': ideal_prediction, 'actual_delta_first_order_prediction': actual_prediction, 'local_linearity_error': error, 'usable': usable, 'ideal_step': ideal, 'seconds': time.perf_counter() - t0, 'memory': mem_snapshot(device), 'delta_reconstruction': {'base_gate_up_sha256': base_hashes[0], 'base_down_sha256': base_hashes[1], 'source_gradient_sha256': source_info['sha256'], 'operation': 'torch BF16 round((base.float32 - eta * gradient.float32)) then subtract base.float32'}}
        rows.append(row)
        log('step_grid_point', domain=domain, **{k: v for k, v in row.items() if k not in ('delta_reconstruction', 'memory')}, sequence_count=32, token_count=8192, memory=row['memory'])
    del ggu, gdn, source
    gc.collect()
    return {'source_gradient_norm': math.sqrt(gnorm2), 'expert_parameter_norm': math.sqrt(theta_norm2), 'base_loss': base_loss, 'loss_floor': floor, 'grid': rows}

def consecutive_window(rows, ideal):
    good = [i for i, row in enumerate(rows) if row['usable'] and (not ideal or row['ideal_step'])]
    runs = []
    current = []
    for i in good:
        if current and i != current[-1] + 1:
            if len(current) >= 3:
                runs.append(current)
            current = []
        current.append(i)
    if len(current) >= 3:
        runs.append(current)
    return runs

def recommendation(steps):
    for ideal, label in ((True, 'IDEAL_T_WINDOW'), (False, 'ACTUAL_DELTA_WINDOW')):
        sets = []
        for domain in ('general', 'code'):
            valid = set((i for run in consecutive_window(steps[domain]['grid'], ideal) for i in run))
            sets.append(valid)
        common = sorted(sets[0] & sets[1])
        runs = []
        current = []
        for i in common:
            if current and i != current[-1] + 1:
                if len(current) >= 3:
                    runs.append(current)
                current = []
            current.append(i)
        if len(current) >= 3:
            runs.append(current)
        if runs:
            run = max(runs, key=len)
            return (label, GRID[run[len(run) // 2]], [GRID[i] for i in run])
    return ('NO_COMMON_WINDOW', None, [])

def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('PYTORCH_ENABLE_MPS_FALLBACK must equal 0')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; BF16 protocol cannot run')
    device = torch.device('mps')
    banks_file = ROOT / 'WP1_Pilot_Banks_v1.json'
    banks = json.loads(banks_file.read_text())
    if banks['model_sha'] != SHA:
        raise RuntimeError('Bank/model SHA mismatch')
    if HfApi().model_info(MODEL, revision=SHA).sha != SHA:
        raise RuntimeError('Model SHA mismatch')
    log('run_start', model=MODEL, sha=SHA, banks_sha256=file_sha(banks_file), python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, huggingface_hub=huggingface_hub.__version__, numpy=np.__version__, mps=True, bf16=True, fallback=os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=SHA, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    freeze_all(model)
    _, mlp, experts = get_target(model)
    if experts.gate_up_proj.dtype != torch.bfloat16 or experts.down_proj.dtype != torch.bfloat16:
        raise RuntimeError('Target expert dtype is not BF16')
    base = (experts.gate_up_proj.detach().clone(), experts.down_proj.detach().clone())
    base_hashes = (tensor_sha(base[0]), tensor_sha(base[1]))
    log('model_loaded', seconds=time.perf_counter() - started, base_hashes=base_hashes, memory=mem_snapshot(device))
    result = {'status': 'RUNNING', 'model': MODEL, 'model_sha': SHA, 'dataset_revisions': banks['dataset_revisions'], 'fallback_events': banks['fallback_events'], 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__, 'numpy': np.__version__, 'device': 'mps', 'dtype': 'bfloat16', 'mps_fallback': '0'}, 'base_expert_hashes': {'gate_up': base_hashes[0], 'down': base_hashes[1]}, 'banks_sha256': file_sha(banks_file), 'gradients': {}, 'steps': {}, 'cross_bank': {}}
    for domain in ('general', 'math', 'science', 'code'):
        result['gradients'][f'{domain}_P'] = collect_bank(model, mlp, experts, banks['banks'][domain]['P'], domain, 'P', device, bootstrap=True)
    discrepancies = [result['gradients'][f'{d}_P']['nested_study'] for d in ('general', 'math', 'science', 'code')]
    candidates = [n for j, n in enumerate(NESTED) if all((study[j]['whole_relative_discrepancy'] <= 0.05 and study[j]['bootstrap_relative_se_gradient_norm'] <= 0.05 for study in discrepancies))]
    result['bank_recommendation'] = candidates[0] if candidates else 'BANK_GT_32'
    for domain in ('general', 'code'):
        result['gradients'][f'{domain}_S'] = collect_bank(model, mlp, experts, banks['banks'][domain]['S'], domain, 'S', device)
    for domain in ('general', 'code'):
        floor = repeated_floor(model, mlp, banks['banks'][domain]['S'], domain, device)
        result['steps'][domain] = step_study(model, mlp, experts, banks['banks'][domain]['S'], domain, result['gradients'][f'{domain}_S'], base, base_hashes, floor, device)
    step_class, candidate_rho, window = recommendation(result['steps'])
    result['step_recommendation'] = {'class': step_class, 'candidate_rho': candidate_rho, 'common_window': window}
    p = {}
    for target in ('general', 'math', 'science', 'code'):
        gt = load_grad(target, 'P')
        p[target] = {}
        for source in ('general', 'code'):
            gs = load_grad(source, 'S')
            p[target][source] = flatten_dot(gt, gs)
            del gs
        del gt
    sealed = ROOT / 'P_coefficients_sealed.json'
    with open(sealed, 'x') as f:
        json.dump({'P_coefficients': p, 'target_bank': 'P32', 'source_bank': 'S32', 'model_sha': SHA}, f, indent=2)
    seal_sha = file_sha(sealed)
    log('P_coefficients_sealed', sha256=seal_sha)
    for domain in ('general', 'math', 'science', 'code'):
        result['gradients'][f'{domain}_E'] = collect_bank(model, mlp, experts, banks['banks'][domain]['E'], domain, 'E', device)
    e = {}
    delta = {}
    for target in ('general', 'math', 'science', 'code'):
        gt = load_grad(target, 'E')
        e[target] = {}
        delta[target] = {}
        for source in ('general', 'code'):
            gs = load_grad(source, 'S')
            e[target][source] = flatten_dot(gt, gs)
            delta[target][source] = e[target][source] - p[target][source]
            del gs
        del gt
    result['cross_bank'] = {'sealed_P_file': sealed.name, 'sealed_P_sha256': seal_sha, 'T_P': p, 'T_E': e, 'delta_T_E_minus_P': delta}
    result['wall_time_s'] = time.perf_counter() - started
    result['final_memory'] = mem_snapshot(device)
    result['formal_cost_estimate'] = {'basis': 'measured pilot bank gradient operations, linear in sequence count', 'P32_seconds_by_domain': {d: result['gradients'][f'{d}_P']['seconds'] for d in ('general', 'math', 'science', 'code')}, 'estimated_target_P_plus_E_seconds_at_recommended_n': None if result['bank_recommendation'] == 'BANK_GT_32' else sum((result['gradients'][f'{d}_P']['seconds'] for d in ('general', 'math', 'science', 'code'))) * 2 * result['bank_recommendation'] / 32}
    result['status'] = 'COMPLETE_PILOT'
    (ROOT / 'WP1_Pilot_Result_v1.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    log('run_complete', seconds=result['wall_time_s'], bank_recommendation=result['bank_recommendation'], step_recommendation=step_class, candidate_rho=candidate_rho)
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log('run_failed', exception=repr(exc))
        raise
