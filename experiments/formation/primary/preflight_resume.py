# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""R0-R2: provenance and amended gate offline; no model/tokenizer/torch calls."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math, numpy as np
from common import *

def run():
    freeze = read(PKG / 'P2J1_AMENDMENT_FREEZE.json')
    package = {}
    for manifest in ['SHA256SUMS_PACKAGE.txt', 'P2J1_AMENDMENT_MANIFEST.sha256']:
        for line in (PKG / manifest).read_text().splitlines():
            h, n = line.split(maxsplit=1)
            assert sha(PKG / n) == h, n
            package[n] = h
    parent = 'Project1_P2J_DeterministicFixedBank_Formation_Protocol_v1_FROZEN.md'
    assert sha(PKG / parent) == freeze['parent_protocol_sha256'] == sha(ROOT / 'work/Project1_P2K_DeterministicFixedBank_Codex_Package_v1' / parent)
    consumed = {}
    for line in (BLOCKED / 'SHA256SUMS_P2J.txt').read_text().splitlines():
        h, n = line.split(maxsplit=1)
        p = BLOCKED / n
        assert sha(p) == h, n
        consumed[str(p)] = h
    for n in ['P2J_EXECUTION_REPORT.md', 'P2J_REPRODUCIBILITY_SOURCE.zip']:
        assert sha(PKG / n) == sha(BLOCKED / n), n
    stop = read(BLOCKED / 'P2J_STOP.json')
    assert stop['protocol_blocker'] == 'P2J_DIMENSIONLESS_STEP_GATE_FAIL' and stop['counters']['PRIMARY_TRAINING_UPDATES'] == 0
    assert not (BLOCKED / 'P2J_PREFLIGHT_SEAL.json').exists()
    audit = read(BLOCKED / 'P2J_REUSE_AUDIT.json')
    for p, h in audit['consumed_sha256'].items():
        assert sha(p) == h, p
        consumed[p] = h
    amp = read(BLOCKED / 'P2J_AMPLITUDE_SEAL.json')
    assert amp['status'] == 'PASS' and amp['selected_rho'] == freeze['selected_rho0'] and (amp['selected_a0'] == freeze['selected_a0'])
    for n in ['P2J_AMPLITUDE_GATE.csv', 'P2J_AMPLITUDE_SEAL.json', 'P2J_ALPHA_GATE.csv', 'P2J_ALPHA_SEAL.json']:
        target = OUT / n
        assert not target.exists()
        target.write_bytes((BLOCKED / n).read_bytes())
        target.chmod(292)
    binding = read(BLOCKED / 'P2J_F0_AND_C_BINDING.json')
    z = np.load(binding['cross_hessian_path'], allow_pickle=False)
    qv = float(np.sum(z['v'][:, :, 0].astype(float) ** 2))
    qu = float(np.sum(z['u'][:, 0].astype(float) ** 2))
    a0 = amp['selected_a0']
    rows = []
    for old in csvread(BLOCKED / 'P2J_ALPHA_GATE.csv'):
        a = float(old['alpha'])
        eps = float(old['epsilon'])
        r = float(old['r_eta'])
        stability = abs(float(old['rho_minus'])) < 1
        total = float(old['actual_step_norm'])
        fraction = float(old['orthogonal_step_fraction'])
        perp = total * fraction
        projected_sq = max(total * total - perp * perp, 0.0)
        parallel = math.sqrt(projected_sq) / (a * a0)
        lo = math.sqrt(projected_sq / max(2 - qv, 2 - qu)) / (a * a0)
        hi = math.sqrt(projected_sq / min(2 - qv, 2 - qu)) / (a * a0)
        passed = stability and (hi <= 0.05 and perp / (a * a0) <= 0.05 if eps == 0 else float(old['one_step_relative_error']) <= 0.05 and fraction <= 0.05)
        rows.append(dict(alpha=a, epsilon=eps, r_eta=r, decaying_branch_stable=stability, original_one_step_relative_error=float(old['one_step_relative_error']), original_orthogonal_step_fraction=fraction, E0_parallel=parallel if eps == 0 else None, E0_parallel_lower_bound=lo if eps == 0 else None, E0_parallel_upper_bound=hi if eps == 0 else None, E0_perp=perp / (a * a0) if eps == 0 else None, criterion='MARGINAL_ABSOLUTE_ALPHA_A0' if eps == 0 else 'ORIGINAL_NONZERO_GATE', passed=passed, model_calls=0))
    assert len(rows) == 45
    passing = [a for a in [0.2, 0.1, 0.05] if all((r['passed'] for r in rows if r['alpha'] == a))]
    chosen = max(passing) if passing else None
    csvout('P2J1_ALPHA_GATE_REEVALUATED.csv', rows)
    write('P2J1_ALPHA_GATE_REEVALUATED.json', dict(status='PASS' if chosen == 0.2 else 'FAIL', selected_alpha=chosen, selected_rho=amp['selected_rho'], a0=a0, model_calls=0, gate_csv_sha256=sha(OUT / 'P2J1_ALPHA_GATE_REEVALUATED.csv'), u1_squared_norm=qu, v1_squared_norm=qv, scalar_reconstruction='projected_norm=sqrt(full_step^2-orthogonal_residual_norm^2); tiny raw-vector norm defect handled with conservative exact bounds; gate evaluated on upper bound', max_E0_parallel=max((r['E0_parallel'] for r in rows if r['epsilon'] == 0)), max_E0_perp=max((r['E0_perp'] for r in rows if r['epsilon'] == 0))))
    if chosen != freeze['selected_alpha']:
        write('P2J1_STOP.json', dict(protocol_blocker='P2J1_AMENDED_ALPHA_GATE_FAIL', counters=COUNTERS))
        return
    pred = []
    s = float(z['s'][0])
    horizons = {}
    for eps in [-0.24, -0.08, 0.0, 0.08, 0.24]:
        for r in [0.25, 1.0, 4.0]:
            B = np.array([[-1.0, 1.0], [r, -r * (1 - eps)]], dtype=float)
            T = np.eye(2) + chosen * B
            vals, vec = np.linalg.eig(T)
            k = int(np.argmax(vals.real))
            e = vec[:, k].real
            e /= np.linalg.norm(e)
            if e[0] < 0:
                e = -e
            rhop = float(vals[k].real)
            rhom = float(vals[1 - k].real)
            g = math.log(abs(rhop))
            N = 0
            if eps:
                N = 4 * math.ceil(math.ceil(math.log(1.25 if eps > 0 else 0.8) / g) / 4)
                horizons[eps, r] = N
            pred.append(dict(cell=len(pred), epsilon=eps, r_eta=r, s1=s, mu=s, lambda_=s * (1 - eps), alpha=chosen, eta_M=chosen / s, eta_U=r * chosen / s, rho_plus=rhop, rho_minus=rhom, gamma_plus=g, e_M=float(e[0]), e_U=float(e[1]), a0=a0, rho0=amp['selected_rho'], N_cell=N, T00=float(T[0, 0]), T01=float(T[0, 1]), T10=float(T[1, 0]), T11=float(T[1, 1]), primary_role='MARGINAL_AUDIT_ONLY' if eps == 0 else 'PRIMARY_NONZERO_CELL'))
    for p in pred:
        if p['epsilon'] == 0:
            p['N_cell'] = max(horizons[-0.08, p['r_eta']], horizons[0.08, p['r_eta']])
        declared = next((x for x in freeze['horizons'] if x['epsilon'] == p['epsilon'] and x['r_eta'] == p['r_eta']))
        assert p['N_cell'] == declared['N_cell'] and abs(p['rho_plus'] - declared['rho_plus']) < 1e-14 and (abs(p['gamma_plus'] - declared['gamma_plus']) < 1e-14)
    assert max((p['N_cell'] for p in pred)) <= 320 and sum((2 * p['N_cell'] for p in pred)) == freeze['total_primary_full_F0_updates_both_signs'] == 912
    csvout('P2J_FROZEN_CELL_PREDICTIONS.csv', pred)
    write('P2J1_PREDICTION_SEAL.json', dict(status='PASS', predictions_sha256=sha(OUT / 'P2J_FROZEN_CELL_PREDICTIONS.csv'), cell_prediction_sha256={str(p['cell']): sha_bytes(p) for p in pred}, total_updates=912))
    write('P2J_REUSE_AUDIT.json', dict(status='EXACT_REUSED_ASSETS_VERIFIED', package_sha256=package, consumed_sha256=consumed, model_dir=audit['model_dir'], model_commit=binding['model_commit'], cross_hessian=binding['cross_hessian_path'], F0_path=binding['F0_file'], hf_cache=_p1_path('path_4f415242142d'), old_blocked_run_preserved=True, counters=COUNTERS, created_utc=utc()))
    write('P2J_F0_AND_C_BINDING.json', dict(binding, amendment_id=freeze['amendment_id'], blocked_binding_sha256=sha(BLOCKED / 'P2J_F0_AND_C_BINDING.json')))
    write('P2J1_PROVENANCE_BINDING.json', dict(parent_protocol_sha256=sha(PKG / parent), amendment_sha256=sha(PKG / 'Project1_P2J1_MarginalCell_Numerical_Amendment_v1_FROZEN.md'), freeze_sha256=sha(PKG / 'P2J1_AMENDMENT_FREEZE.json'), blocked_report_sha256=sha(BLOCKED / 'P2J_EXECUTION_REPORT.md'), blocked_source_package_sha256=sha(BLOCKED / 'P2J_REPRODUCIBILITY_SOURCE.zip'), blocked_amplitude_gate_sha256=sha(BLOCKED / 'P2J_AMPLITUDE_GATE.csv'), blocked_alpha_gate_sha256=sha(BLOCKED / 'P2J_ALPHA_GATE.csv'), primary_updates_before_amendment=0, primary_decisions_before_amendment=0, amendment_count=1))
    write('P2J_PREFLIGHT_SEAL.json', dict(status='PASS', amended=True, amendment_id=freeze['amendment_id'], selected_rho=amp['selected_rho'], selected_a0=a0, selected_alpha=chosen, artifact_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}, source_sha256={str(p): sha(p) for p in SOURCE.glob('*.py')}, sealed_utc=utc(), counters=COUNTERS))
    print('R0-R2 PASS: offline gate alpha=.20; 15 exact horizons verified; budget 912; primary updates before seal 0', flush=True)

def sha_bytes(p):
    import hashlib, json
    return hashlib.sha256(json.dumps(p, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if __name__ == '__main__':
    run()
