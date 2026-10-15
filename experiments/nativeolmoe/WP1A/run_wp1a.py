# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen WP1-A source self-calibration. No cross-source or P/E computation."""
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
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from validated_preflight_v1p1 import freeze_all, get_target, mem_snapshot, sync
from validated_pilot_run import accumulate, empty_grad, probe_one
HERE = Path(__file__).resolve().parent
PROJECT = Path(_p1_path('path_7e3390fd04bd'))
BANK_PATH = PROJECT / 'formal_prep_v1p2' / 'Formal_Banks_v1p2.json'
VERIFY_PATH = PROJECT / 'formal_prep_v1p2' / 'Formal_Prep_Verification_v1p2.json'
BANK_SHA = '80eac9dd99453359aa38a10c2b726bf4d21dc67e708682aa5ca3f0b68db8727e'
MODEL = 'allenai/OLMoE-1B-7B-0924'
MODEL_SHA = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
RHO = 2.0 ** (-13)
SCALES = (0.5, 1.0, 2.0)
RESULT = HERE / 'WP1A_Formal_Result_v1.json'
LOSSES = HERE / 'WP1A_PerDocumentLosses_v1.json'
MANIFEST = HERE / 'WP1A_SourceGradient_Manifest_v1.json'
SUMMARY = HERE / 'WP1A_Formal_Summary_v1.md'
LOG = HERE / 'WP1A_Run.log'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')

def log(event, **details):
    line = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **details}
    with LOG.open('a') as f:
        f.write(json.dumps(line, sort_keys=True, allow_nan=False) + '\n')
    print(event, details, flush=True)

def check_frozen_inputs():
    observed_sha = sha256(BANK_PATH)
    if observed_sha != BANK_SHA:
        raise RuntimeError(f'Formal bank SHA mismatch: {observed_sha}')
    verification = json.loads(VERIFY_PATH.read_text())
    if verification.get('status') != 'READY' or not all(verification.get('checks', {}).values()):
        raise RuntimeError('Formal preparation verification is not READY')
    banks = json.loads(BANK_PATH.read_text())
    if banks.get('model') != MODEL or banks.get('model_sha') != MODEL_SHA:
        raise RuntimeError('Frozen bank model identity mismatch')
    if banks.get('sequence_length') != 256 or tuple(banks['banks']) != DOMAINS:
        raise RuntimeError('Formal bank structure mismatch')
    for domain in DOMAINS:
        source = banks['banks'][domain]['S']
        if len(source) != 96 or any((len(row['token_ids']) != 256 for row in source)):
            raise RuntimeError(f'S96 length mismatch: {domain}')
    return banks

def check_environment():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('PYTORCH_ENABLE_MPS_FALLBACK must equal 0')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS is unavailable')
    return torch.device('mps')

def selected_names(model, experts):
    selected = {id(experts.gate_up_proj), id(experts.down_proj)}
    names = {id(p): name for name, p in model.named_parameters() if id(p) in selected}
    if set(names) != selected or len(names) != 2:
        raise RuntimeError('Exactly two final-layer expert parameters were not found')
    return (names[id(experts.gate_up_proj)], names[id(experts.down_proj)])

def assert_no_parameter_gradients(model):
    offenders = [name for name, p in model.named_parameters() if p.grad is not None]
    if offenders:
        raise RuntimeError(f'Model parameters received gradients: {offenders[:8]}')

def assert_restored(experts, base):
    if not (torch.equal(experts.gate_up_proj.detach(), base[0]) and torch.equal(experts.down_proj.detach(), base[1])):
        raise RuntimeError('Exact base expert checkpoint restoration failed')

def eval_s96(model, bank, device):
    losses = []
    for row in bank:
        x = torch.tensor([row['token_ids']], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(input_ids=x, labels=x, use_cache=False)
        losses.append(float(output.loss.detach().cpu()))
        del x, output
    sync(device)
    if len(losses) != 96 or not all((math.isfinite(v) for v in losses)):
        raise RuntimeError('Invalid per-document S96 losses')
    return losses

def norm_per_expert(grad_gu, grad_dn):
    return [math.sqrt(float(np.einsum('ij,ij->', grad_gu[i], grad_gu[i], dtype=np.float64)) + float(np.einsum('ij,ij->', grad_dn[i], grad_dn[i], dtype=np.float64))) for i in range(grad_gu.shape[0])]

def collect_gradient(model, mlp, experts, bank, domain, device):
    t0 = time.perf_counter()
    acc = empty_grad(experts)
    loads_total = np.zeros(int(experts.gate_up_proj.shape[0]), dtype=np.int64)
    gradient_pass_losses = []
    for index, row in enumerate(bank, 1):
        grads, loads, loss = probe_one(model, mlp, experts, row['token_ids'], device)
        assert_no_parameter_gradients(model)
        accumulate(acc, grads)
        loads_total += loads
        gradient_pass_losses.append(loss)
        if index % 12 == 0:
            log('gradient_progress', domain=domain, documents=index, seconds=time.perf_counter() - t0)
    mean = (acc[0] / 96.0, acc[1] / 96.0)
    if not all((np.isfinite(x).all() for x in mean)):
        raise RuntimeError(f'Nonfinite source gradient: {domain}')
    del acc
    gc.collect()
    return (mean, gradient_pass_losses, loads_total, time.perf_counter() - t0)

def save_gradient(domain, names, mean):
    path = HERE / f'WP1A_SourceGradient_{domain}_v1.safetensors'
    tensors = {names[0]: torch.from_numpy(mean[0].copy()), names[1]: torch.from_numpy(mean[1].copy())}
    if any((t.dtype != torch.float32 for t in tensors.values())):
        raise RuntimeError('Gradient artifact is not FP32')
    save_file(tensors, str(path), metadata={'domain': domain, 'bank': 'S96', 'model_sha': MODEL_SHA})
    return {'path': path.name, 'sha256': sha256(path), 'tensor_names': list(tensors), 'tensor_shapes': {k: list(t.shape) for k, t in tensors.items()}, 'dtype': 'float32', 'source_documents': 96}

def scalar_audit(base, updated, grad, eta, c, g2, theta2):
    with torch.no_grad():
        dgu = updated[0] - base[0]
        ddn = updated[1] - base[1]
        ideal_gu = -c * eta * grad[0]
        ideal_dn = -c * eta * grad[1]
        d2 = float((dgu.square().sum() + ddn.square().sum()).cpu())
        r2 = float(((dgu - ideal_gu).square().sum() + (ddn - ideal_dn).square().sum()).cpu())
        dot = float(((dgu * grad[0]).sum() + (ddn * grad[1]).sum()).cpu())
    ideal_norm = c * eta * math.sqrt(g2)
    return {'actual_relative_displacement': math.sqrt(d2 / theta2), 'd_quant': math.sqrt(r2) / max(ideal_norm, 1e-30), 'cosine_alignment': -dot / max(math.sqrt(d2 * g2), 1e-30), 'actual_displacement_first_order_prediction': dot}

def run_source(model, mlp, experts, base, names, bank, domain, device, theta2, per_doc, manifest):
    assert_restored(experts, base)
    mean, gradient_pass_losses, loads, gradient_seconds = collect_gradient(model, mlp, experts, bank, domain, device)
    artifact = save_gradient(domain, names, mean)
    manifest[domain] = artifact
    dump(MANIFEST, {'schema': 'WP1A_SourceGradient_Manifest_v1', 'model_sha': MODEL_SHA, 'bank_sha256': BANK_SHA, 'gradients': manifest})
    log('gradient_complete', domain=domain, seconds=gradient_seconds, artifact=artifact['path'], sha256=artifact['sha256'])
    grad = (torch.from_numpy(mean[0]).to(device), torch.from_numpy(mean[1]).to(device))
    with torch.no_grad():
        g2 = float((grad[0].square().sum() + grad[1].square().sum()).cpu())
    if not math.isfinite(g2) or g2 <= 0:
        raise RuntimeError(f'Invalid source gradient norm: {domain}')
    eta = RHO * math.sqrt(theta2 / g2)
    base_losses = eval_s96(model, bank, device)
    base_loss = statistics.mean(base_losses)
    source_losses = {'source_id': [row['source_id'] for row in bank], 'gradient_pass': gradient_pass_losses, 'base': base_losses, 'c_0p5': None, 'c_1': None, 'c_2': None}
    per_doc[domain] = source_losses
    dump(LOSSES, {'schema': 'WP1A_PerDocumentLosses_v1', 'loss_definition': 'mean NLL over 255 next-token positions; equal document weighting', 'domains': per_doc})
    steps = {}
    for c, key in ((0.5, 'c_0p5'), (1.0, 'c_1'), (2.0, 'c_2')):
        with torch.no_grad():
            experts.gate_up_proj.copy_(base[0])
            experts.down_proj.copy_(base[1])
        assert_restored(experts, base)
        try:
            with torch.no_grad():
                updated = (base[0] - c * eta * grad[0], base[1] - c * eta * grad[1])
                experts.gate_up_proj.copy_(updated[0])
                experts.down_proj.copy_(updated[1])
            audit = scalar_audit(base, updated, grad, eta, c, g2, theta2)
            values = eval_s96(model, bank, device)
            source_losses[key] = values
            loss = statistics.mean(values)
            steps[key] = {'scale': c, 'loss': loss, 'delta_loss': loss - base_loss, **audit}
            log('step_complete', domain=domain, scale=c, loss=loss, delta_loss=loss - base_loss, d_quant=audit['d_quant'], cosine_alignment=audit['cosine_alignment'])
            dump(LOSSES, {'schema': 'WP1A_PerDocumentLosses_v1', 'loss_definition': 'mean NLL over 255 next-token positions; equal document weighting', 'domains': per_doc})
        finally:
            with torch.no_grad():
                experts.gate_up_proj.copy_(base[0])
                experts.down_proj.copy_(base[1])
            assert_restored(experts, base)
        del updated
    assert_no_parameter_gradients(model)
    delta = steps['c_1']['delta_loss']
    ideal_prediction = -eta * g2
    actual_prediction = steps['c_1']['actual_displacement_first_order_prediction']
    e_ideal = abs(delta - ideal_prediction) / max(abs(ideal_prediction), 1e-12)
    e_actual = abs(delta - actual_prediction) / max(abs(actual_prediction), 1e-12)
    x = np.array([[c, c * c] for c in SCALES], dtype=np.float64)
    y = np.array([steps[k]['delta_loss'] for k in ('c_0p5', 'c_1', 'c_2')], dtype=np.float64)
    a1, a2 = np.linalg.lstsq(x, y, rcond=None)[0]
    per_expert = norm_per_expert(*mean)
    numerical_pass = steps['c_1']['d_quant'] <= 0.001 and steps['c_1']['cosine_alignment'] >= 0.99999
    improvement = -delta / base_loss
    result = {'source': domain, 'bank': 'S96', 'base_loss': base_loss, 'gradient_pass_base_loss': statistics.mean(gradient_pass_losses), 'source_gradient_norm': math.sqrt(g2), 'expert_parameter_norm': math.sqrt(theta2), 'per_expert_gradient_norms': per_expert, 'active_expert_count': int(np.count_nonzero(loads)), 'token_expert_assignments': loads.tolist(), 'total_assignments': int(loads.sum()), 'rho': RHO, 'eta': eta, 'steps': steps, 'ideal_first_order_prediction': ideal_prediction, 'actual_displacement_first_order_prediction': actual_prediction, 'primary_first_order_relative_error': e_ideal, 'actual_displacement_first_order_relative_error': e_actual, 'quadratic_zero_intercept': {'a1': float(a1), 'a2': float(a2), 'linear_coefficient_relative_error': float(abs(a1 + eta * g2) / (eta * g2)), 'curvature_ratio': float(abs(a2) / (abs(a1) + np.finfo(np.float64).eps)), 'epsilon': float(np.finfo(np.float64).eps)}, 'relative_loss_decline': improvement, 'gradient_artifact': artifact, 'gradient_seconds': gradient_seconds, 'displacement_pass': numerical_pass, 'loss_improved': delta < 0, 'first_order_error_pass': e_ideal <= 0.1, 'checkpoint_restoration_pass': True, 'nonselected_gradient_pass': True}
    log('source_complete', domain=domain, relative_loss_decline=improvement, displacement_pass=numerical_pass, first_order_error=e_ideal)
    del mean, grad
    gc.collect()
    return result

def summary(result):
    lines = ['# WP1-A Formal Source Self-Calibration v1', '', f'Model: `{MODEL}` at `{MODEL_SHA}`. Frozen S96 bank SHA256: `{BANK_SHA}`.', f'Numerics: FP32 on Apple MPS; `PYTORCH_ENABLE_MPS_FALLBACK=0`; rho = {RHO}.', '']
    if result['sources']:
        lines += ['| Source | Base loss | Delta loss c=1 | d_quant | Cosine | First-order error | Relative decline |', '|---|---:|---:|---:|---:|---:|---:|']
        for domain, row in result['sources'].items():
            step = row['steps']['c_1']
            lines.append(f"| {domain} | {row['base_loss']:.8g} | {step['delta_loss']:.8g} | {step['d_quant']:.5g} | {step['cosine_alignment']:.8g} | {row['primary_first_order_relative_error']:.5g} | {row['relative_loss_decline']:.5g} |")
        lines.append('')
        lines.append(f"Mean relative loss decline: {result.get('mean_relative_loss_decline', float('nan')):.8g}.")
    if result.get('error'):
        lines += ['', 'Implementation error: `' + result['error'] + '`']
    lines += ['', f"WP1A_STATUS = {result['WP1A_STATUS']}"]
    SUMMARY.write_text('\n'.join(lines) + '\n')

def finalize(result):
    dump(RESULT, result)
    summary(result)
    files = sorted((p for p in HERE.iterdir() if p.is_file() and p.name != 'SHA256SUMS'))
    (HERE / 'SHA256SUMS').write_text(''.join((f'{sha256(p)}  {p.name}\n' for p in files)))

def main():
    started = time.perf_counter()
    result = {'schema': 'WP1A_Formal_Result_v1', 'WP1A_STATUS': 'IMPLEMENTATION_BLOCKED', 'model': MODEL, 'model_sha': MODEL_SHA, 'bank_sha256': BANK_SHA, 'rho': RHO, 'device': 'mps', 'dtype': 'float32', 'sources': {}, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'platform': platform.platform()}}
    per_doc, manifest = ({}, {})
    try:
        banks = check_frozen_inputs()
        device = check_environment()
        log('inputs_verified', bank_sha256=BANK_SHA, model_sha=MODEL_SHA)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_SHA, dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True)
        if getattr(model.config, '_commit_hash', MODEL_SHA) != MODEL_SHA:
            raise RuntimeError('Loaded model commit differs from pinned SHA')
        model.to(device)
        model.eval()
        model.config.use_cache = False
        freeze_all(model)
        if not all((p.dtype == torch.float32 and p.device.type == 'mps' and (not p.requires_grad) for p in model.parameters())):
            raise RuntimeError('Model parameter dtype/device/frozen check failed')
        _, mlp, experts = get_target(model)
        names = selected_names(model, experts)
        base = (experts.gate_up_proj.detach().clone(), experts.down_proj.detach().clone())
        with torch.no_grad():
            theta2 = float((base[0].square().sum() + base[1].square().sum()).cpu())
        if theta2 <= 0 or not math.isfinite(theta2):
            raise RuntimeError('Invalid selected expert parameter norm')
        result['selected_parameters'] = {names[0]: list(base[0].shape), names[1]: list(base[1].shape)}
        result['expert_parameter_norm'] = math.sqrt(theta2)
        log('model_loaded', selected_parameters=result['selected_parameters'], memory=mem_snapshot(device))
        for domain in DOMAINS:
            log('source_start', domain=domain)
            result['sources'][domain] = run_source(model, mlp, experts, base, names, banks['banks'][domain]['S'], domain, device, theta2, per_doc, manifest)
            dump(RESULT, result)
        improvements = [result['sources'][d]['relative_loss_decline'] for d in DOMAINS]
        result['mean_relative_loss_decline'] = statistics.mean(improvements)
        result['gate'] = {'inputs_model_precision_device': True, 'no_nonselected_gradients': all((result['sources'][d]['nonselected_gradient_pass'] for d in DOMAINS)), 'realized_displacement': all((result['sources'][d]['displacement_pass'] for d in DOMAINS)), 'all_source_losses_improved': all((result['sources'][d]['loss_improved'] for d in DOMAINS)), 'first_order_error': all((result['sources'][d]['first_order_error_pass'] for d in DOMAINS)), 'mean_relative_decline': result['mean_relative_loss_decline'] >= 0.001, 'exact_restoration': all((result['sources'][d]['checkpoint_restoration_pass'] for d in DOMAINS))}
        result['WP1A_STATUS'] = 'PASS' if all(result['gate'].values()) else 'FAIL'
        log('run_complete', status=result['WP1A_STATUS'], seconds=time.perf_counter() - started, mean_relative_loss_decline=result['mean_relative_loss_decline'])
    except Exception as exc:
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        result['WP1A_STATUS'] = 'IMPLEMENTATION_BLOCKED'
        log('run_blocked', error=repr(exc), traceback=result['traceback'])
    finally:
        finalize(result)
if __name__ == '__main__':
    main()
