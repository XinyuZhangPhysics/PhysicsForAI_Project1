# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Engineering only on the inherited, already-open G112 diagnostic subset."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import copy, inspect, time
from engine import *

def g112():
    diag = read(PILOT / 'outputs_v151/diagnostic_manifest_14.json')
    path = Path(diag['bank_manifest_path'])
    if sha(path) != diag['bank_manifest_sha256']:
        raise RuntimeError('G112_MANIFEST_HASH_FAILURE')
    manifest = read(path)
    byid = {r['document_id']: r for r in manifest['documents']}
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    docs = []
    identities = []
    for selected in diag['documents']:
        row = byid[selected['document_id']]
        p = (path.parent / row['text_or_token_path']).resolve()
        if p.suffix == '.npy':
            ids = np.load(p, allow_pickle=False).tolist()
        elif p.suffix == '.json':
            ids = read(p)
        else:
            ids = tokenizer(p.read_text(), add_special_tokens=manifest['preprocessing']['add_special_tokens'])['input_ids']
        if len(ids) != 256 or not all((type(t) is int for t in ids)):
            raise RuntimeError('G112_TOKEN_CONTRACT_FAILURE')
        docs.append(ids)
        identities.append(dict(document_id=row['document_id'], **info(p)))
    if len(docs) != 14:
        raise RuntimeError('G112_SUBSET_COUNT_FAILURE')
    return (docs, dict(manifest=info(path), diagnostic_manifest=info(PILOT / 'outputs_v151/diagnostic_manifest_14.json'), documents=identities))

def main():
    frozen()
    seed()
    docs, identities = g112()
    start = time.monotonic()
    event('G112_PREFLIGHT_START', Q3_bank_forwards=0)
    model = load_model()
    before = {n: p._version for n, p in model.named_parameters()}
    native = []
    with torch.no_grad():
        for ids in docs:
            native.append(float(nll(model, ids).cpu()))
    baseline = Adapter(model, 'BASELINE')
    baseline_values = []
    with torch.no_grad():
        baseline.alpha[7].fill_(0.02)
        baseline.alpha[15].fill_(-0.02)
        for ids in docs:
            baseline_values.append(float(nll(model, ids).cpu()))
    baseline.close()
    discrepancy = max((abs(a - b) for a, b in zip(native, baseline_values)))
    if discrepancy != 0:
        raise RuntimeError('BASELINE_FUNCTIONAL_INVARIANCE_FAILURE')
    replays = []
    gradchecks = []
    for replay in range(2):
        seed()
        adapter = Adapter(model, 'CANDIDATE')
        optimizer = adapter.optimizer()
        optimizer.zero_grad(set_to_none=True)
        values = []
        for i, ids in enumerate(docs[:8]):
            loss = nll(model, ids)
            values.append(float(loss.detach().cpu()))
            (loss / 8).backward()
            grads = [float(p.grad.detach().cpu()) for p in adapter.alpha.values()]
            if not all(np.isfinite(grads)):
                raise RuntimeError('NONFINITE_ALPHA_GRADIENT')
            gradchecks.append(dict(replay=replay, document=i, gradients=grads))
        optimizer.step()
        adapter.project()
        replays.append(dict(losses=values, gradients=grads, alphas=adapter.values()))
        adapter.close()
        del optimizer, adapter
    for key in ['losses', 'gradients']:
        if not np.allclose(replays[0][key], replays[1][key], rtol=3e-06, atol=0):
            raise RuntimeError('DETERMINISTIC_REPLAY_FAILURE:' + key)
    if replays[0]['alphas'] != replays[1]['alphas']:
        raise RuntimeError('DETERMINISTIC_ALPHA_REPLAY_FAILURE')
    adapter = Adapter(model, 'CANDIDATE')
    with torch.no_grad():
        adapter.alpha[7].fill_(0.04)
        adapter.alpha[15].fill_(-0.04)
    adapter.project()
    if not all((abs(v) <= 0.020000001 for v in adapter.values().values())):
        raise RuntimeError('PROJECTION_FAILURE')
    semantic = []
    handles = []
    for layer in (7, 15):
        gate = model.model.layers[layer].mlp.gate

        def pre(module, args, l=layer):
            native_z = F.linear(args[0], module.weight, module.bias)
            semantic.append(dict(layer=l, native=native_z.detach()))

        def post(module, args, output, l=layer):
            native_z = next((r['native'] for r in reversed(semantic) if r['layer'] == l))
            expected = native_z.float() + adapter.alpha[l] * adapter.bias[l]
            semantic[-1]['max_error'] = float((output - expected).abs().max().detach().cpu())
        handles.append(gate.register_forward_pre_hook(pre))
        handles.append(gate.register_forward_hook(post))
    with torch.no_grad():
        value = float(nll(model, docs[0]).cpu())
    for h in handles:
        h.remove()
    errors = [r['max_error'] for r in semantic]
    if len(errors) != 2 or max(errors) != 0:
        raise RuntimeError('ADDITIVE_PRE_TOPK_FAILURE')
    state = adapter.values()
    path = ROOT / 'preflight_alpha.json'
    write(path, state)
    adapter.close()
    adapter = Adapter(model, 'CANDIDATE')
    with torch.no_grad():
        for l, p in adapter.alpha.items():
            p.copy_(torch.tensor(read(path)[f'alpha{l}'], device='mps'))
        reloaded = float(nll(model, docs[0]).cpu())
    if value != reloaded:
        raise RuntimeError('SERIALIZATION_RELOAD_FAILURE')
    audit = base_audit(model, before)
    if any(audit.values()):
        raise RuntimeError('BASE_MODEL_MUTATION_OR_GRADIENT_FAILURE')
    adapter.close()
    result = dict(status='PASS', time_UTC=now(), scientific_freeze_sha256=sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'), scope='G112_ONLY', Q3_bank_forwards=0, source_identities=identities, finite_alpha_gradients=True, gradient_checks=gradchecks, additive_pre_topk_semantics=True, pre_topk_max_errors=errors, projection=True, deterministic_replays=replays, relative_tolerance=3e-06, baseline_NLL_max_absolute_difference=discrepancy, baseline_functional_invariance=True, serialization_reload=True, memory_feasibility=True, MPS_current_allocated_bytes=torch.mps.current_allocated_memory(), MPS_driver_allocated_bytes=torch.mps.driver_allocated_memory(), elapsed_seconds=time.monotonic() - start, **audit)
    write(BUNDLE / 'engineering_preflight_audit.json', result)
    event('G112_PREFLIGHT_PASS', audit_sha256=sha(BUNDLE / 'engineering_preflight_audit.json'))
    print('G112 PREFLIGHT PASS; no Q3 bank model access', flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        safe = str(e) if isinstance(e, (AssertionError, RuntimeError)) and len(str(e)) < 2000 else 'ENGINEERING_OPERATION_FAILURE'
        write(BUNDLE / 'technical_stop_trigger.json', dict(stage='D_G112_PREFLIGHT', exception_class=type(e).__name__, reason=safe, time_UTC=now()))
        write(BUNDLE / 'engineering_preflight_audit.json', dict(status='FAIL', scope='G112_ONLY', Q3_bank_forwards=0, reason=safe))
        event('TECHNICAL_STOP', stage='D_G112_PREFLIGHT', exception_class=type(e).__name__)
        print('TECHNICAL STOP: G112 preflight failed; see saved redacted audit', flush=True)
        sys.exit(2)
