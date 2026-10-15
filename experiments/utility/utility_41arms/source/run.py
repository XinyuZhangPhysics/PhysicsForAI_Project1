# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Exactly 41 frozen arms, with heldout access after each final adaptation seal."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv, time
from engine import *
from firewall import actual_firewall

def main():
    freeze = frozen()
    if (BUNDLE / 'technical_stop_trigger.json').exists():
        raise RuntimeError('RUN_LOCKED_BY_TECHNICAL_STOP')
    if read(BUNDLE / 'engineering_preflight_audit.json')['status'] != 'PASS':
        raise RuntimeError('PREFLIGHT_NOT_PASS')
    fw, parts = actual_firewall()
    adapt = ordered(parts['ADAPT'])
    heldout = ordered(parts['HELDOUT'])
    for arm in ARMS:
        folder = BUNDLE / 'arms' / arm
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'completion_receipt.json').exists():
            raise RuntimeError('FRESH_ARM_REQUIRED')
        seed()
        model = load_model()
        before = {n: p._version for n, p in model.named_parameters()}
        adapter = Adapter(model, arm)
        optimizer = adapter.optimizer()
        optimizer.zero_grad(set_to_none=True)
        assert adapter.values() == dict(alpha7=0, alpha15=0)
        start = time.monotonic()
        steps = 0
        processed = 0
        boundary = 0
        event('ARM_START', arm=arm, base_sha=MODEL_SHA, alpha7=0, alpha15=0)
        for epoch in range(2):
            for i, row in enumerate(adapt):
                ids = fw.load_adapt(row['document_id'], arm)
                loss = nll(model, ids)
                (loss / 8).backward()
                processed += 1
                if (i + 1) % 8 == 0:
                    if any((p.grad is None or not bool(torch.isfinite(p.grad)) for p in adapter.alpha.values())):
                        raise RuntimeError('NONFINITE_ALPHA_GRADIENT')
                    optimizer.step()
                    adapter.project()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    boundary += int(any((abs(v) >= 0.019999999 for v in adapter.values().values())))
                    if steps % 7 == 0:
                        print(f'{arm} adaptation steps {steps}/56', flush=True)
        assert steps == 56 and processed == 448
        audit = base_audit(model, before)
        if any(audit.values()):
            raise RuntimeError('BASE_MODEL_MUTATION_OR_GRADIENT_FAILURE')
        write(folder / 'final_alpha.json', dict(arm=arm, **adapter.values()))
        receipt = dict(status='SEALED_FINAL', arm=arm, time_UTC=now(), scientific_freeze_sha256=sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'), state_sha256=sha(folder / 'final_alpha.json'), epochs_completed=2, adapt_documents_processed=processed, optimizer_steps=steps, steps_touching_boundary=boundary, elapsed_adaptation_seconds=time.monotonic() - start, **audit)
        write(folder / 'adaptation_receipt.json', receipt)
        (folder / 'adaptation_receipt.sha256').write_text(sha(folder / 'adaptation_receipt.json') + '  adaptation_receipt.json\n')
        for p in ['final_alpha.json', 'adaptation_receipt.json', 'adaptation_receipt.sha256']:
            (folder / p).chmod(292)
        event('ARM_ADAPTER_SEALED', arm=arm, state_sha256=sha(folder / 'final_alpha.json'), receipt_sha256=sha(folder / 'adaptation_receipt.json'))
        values = []
        with torch.no_grad():
            for i, row in enumerate(heldout):
                ids = fw.load_heldout(row['document_id'], arm)
                loss = float(nll(model, ids).cpu())
                values.append(dict(arm=arm, document_id=row['document_id'], domain=row['domain'], nll=loss, predicted_tokens=255))
                if (i + 1) % 112 == 0:
                    print(f'{arm} heldout documents {i + 1}/448', flush=True)
        write(folder / 'heldout_nll.json', values)
        audit = base_audit(model, before)
        if any(audit.values()):
            raise RuntimeError('BASE_MODEL_MUTATION_OR_GRADIENT_FAILURE')
        write(folder / 'completion_receipt.json', dict(status='COMPLETE', arm=arm, time_UTC=now(), scientific_freeze_sha256=sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'), adaptation_receipt_sha256=sha(folder / 'adaptation_receipt.json'), nll_table_sha256=sha(folder / 'heldout_nll.json'), heldout_documents=448, elapsed_arm_seconds=time.monotonic() - start, **audit))
        (folder / 'completion_receipt.sha256').write_text(sha(folder / 'completion_receipt.json') + '  completion_receipt.json\n')
        for p in ['heldout_nll.json', 'completion_receipt.json', 'completion_receipt.sha256']:
            (folder / p).chmod(292)
        event('ARM_COMPLETE', arm=arm, completion_receipt_sha256=sha(folder / 'completion_receipt.json'))
        print(f'{arm} COMPLETE', flush=True)
        adapter.close()
        del optimizer, adapter, model
        gc.collect()
        torch.mps.empty_cache()
    with (BUNDLE / 'heldout_nll.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'document_id', 'domain', 'nll', 'predicted_tokens'])
        w.writeheader()
        for arm in ARMS:
            w.writerows(read(BUNDLE / 'arms' / arm / 'heldout_nll.json'))
    event('ALL_41_ARMS_COMPLETE', heldout_values=18368, table_sha256=sha(BUNDLE / 'heldout_nll.csv'))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        safe = str(e) if isinstance(e, (AssertionError, RuntimeError)) and len(str(e)) < 2000 else 'ARM_EXECUTION_OPERATION_FAILURE'
        write(BUNDLE / 'technical_stop_trigger.json', dict(stage='E_ARMS', exception_class=type(e).__name__, reason=safe, time_UTC=now()))
        event('TECHNICAL_STOP', stage='E_ARMS', exception_class=type(e).__name__)
        print('TECHNICAL STOP: arm execution failure; see redacted audit', flush=True)
        sys.exit(2)
