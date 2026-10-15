# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""M0/M1: exact offline asset audit and theory-only prediction seal."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, socket, math
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

def deny(*a, **k):
    raise RuntimeError('P2M_NETWORK_PROHIBITED')
socket.create_connection = deny
socket.socket.connect = deny
socket.socket.connect_ex = deny
from common import *
import numpy as np
from safetensors import safe_open

def run():
    freeze = read(PKG / 'P2L_PROTOCOL_FREEZE.json')
    consumed = {}
    package = {}
    for manifest in ['SHA256SUMS_PACKAGE.txt', 'P2L_PROTOCOL_MANIFEST.sha256']:
        for line in (PKG / manifest).read_text().splitlines():
            h, n = line.split(maxsplit=1)
            assert sha(PKG / n) == h, n
            package[str(PKG / n)] = h
        package[str(PKG / manifest)] = sha(PKG / manifest)
    js = read(J / 'P2J_PREFLIGHT_SEAL.json')
    assert js['status'] == 'PASS'
    names = ['P2J_AMPLITUDE_SEAL.json', 'P2J1_ALPHA_GATE_REEVALUATED.json', 'P2J1_ALPHA_GATE_REEVALUATED.csv', 'P2J1_PROVENANCE_BINDING.json', 'P2J_F0_AND_C_BINDING.json', 'P2J_REUSE_AUDIT.json']
    for n in names:
        assert sha(J / n) == js['artifact_sha256'][n], n
        consumed[str(J / n)] = sha(J / n)
    consumed[str(J / 'P2J_PREFLIGHT_SEAL.json')] = sha(J / 'P2J_PREFLIGHT_SEAL.json')
    amp = read(J / names[0])
    alpha = read(J / names[1])
    binding = read(J / 'P2J_F0_AND_C_BINDING.json')
    audit = read(J / 'P2J_REUSE_AUDIT.json')
    assert amp['status'] == 'PASS' and amp['selected_rho'] == freeze['rho0'] == 0.0001
    assert alpha['status'] == 'PASS' and alpha['selected_alpha'] == js['selected_alpha'] == freeze['alpha'] == 0.2
    assert js['selected_rho'] == amp['selected_rho'] and js['selected_a0'] == amp['selected_a0'] == alpha['a0']
    assert js['counters']['PRIMARY_MULTISTEP_UPDATES_BEFORE_AMENDED_SEAL'] == 0
    assert binding['model_commit'] == audit['model_commit'] == freeze['model_commit']
    assert binding['layer_index_zero_based'] == freeze['layer_zero_based'] == 20
    for p, h in audit['consumed_sha256'].items():
        assert sha(p) == h, p
        consumed[p] = h
    modeldir = Path(audit['model_dir'])
    assert modeldir.name == freeze['model_commit']
    modelhashes = {}
    metadata = read(modeldir / 'MODEL_ASSET.json')
    assert metadata['revision'] == freeze['model_commit'] and metadata['repo'] == freeze['model']
    for p, h in metadata['file_hashes'].items():
        assert consumed[p] == h
    for p in modeldir.iterdir():
        if p.is_file():
            if p.name == 'MODEL_ASSET.json':
                consumed[str(p)] = sha(p)
            assert str(p) in consumed, p
            modelhashes[str(p)] = consumed[str(p)]
    hseal = read(OLD / 'outputs/P2H_PREFLIGHT_SEAL.json')
    for n in ['P2H_DATA_BANK_SEAL.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz']:
        p = OLD / 'outputs' / n
        assert sha(p) == hseal['artifact_sha256'][n]
        consumed[str(p)] = sha(p)
    consumed[str(OLD / 'outputs/P2H_PREFLIGHT_SEAL.json')] = sha(OLD / 'outputs/P2H_PREFLIGHT_SEAL.json')
    bankseal = read(OLD / 'outputs/P2H_DATA_BANK_SEAL.json')['banks']['F0']
    f0 = np.load(binding['F0_file'], allow_pickle=False)
    assert list(f0.shape) == bankseal['shape'] == [32, 256] and str(bankseal['path']) == binding['F0_file']
    assert sha(binding['F0_file']) == bankseal['file_sha256'] == binding['F0_file_sha256']
    assert array_sha(f0) == bankseal['token_array_sha256'] == binding['F0_array_sha256']
    for row, rec in zip(f0, bankseal['sequences']):
        assert array_sha(row) == rec['token_array_sha256']
    z = np.load(binding['cross_hessian_path'], allow_pickle=False)
    s = float(z['s'][0])
    u = z['u'][:, 0].copy()
    v = z['v'][:, :, 0].copy()
    assert sha(binding['cross_hessian_path']) == binding['cross_hessian_sha256']
    assert s == binding['s1'] and array_sha(u) == binding['u1_array_sha256'] and (array_sha(v) == binding['v1_array_sha256'])
    assert u.shape == (576,) and v.shape == (576, 1536) and (abs(np.linalg.norm(u.astype(float)) - 1) < 1e-06) and (abs(np.linalg.norm(v.astype(float)) - 1) < 1e-06)
    with safe_open(modeldir / 'model.safetensors', framework='pt', device='cpu') as f:
        w = f.get_tensor('model.layers.20.mlp.down_proj.weight')
        wd = float(w.double().norm())
    a0 = freeze['rho0'] * wd
    assert math.isclose(a0, amp['selected_a0'], rel_tol=1e-14)
    amendment = ROOT.parent / 'jie-2/work/Project1_P2J1_Resume_Codex_Package_v1'
    prov = read(J / 'P2J1_PROVENANCE_BINDING.json')
    for n, key in [('P2J1_AMENDMENT_FREEZE.json', 'freeze_sha256'), ('Project1_P2J1_MarginalCell_Numerical_Amendment_v1_FROZEN.md', 'amendment_sha256'), ('Project1_P2J_DeterministicFixedBank_Formation_Protocol_v1_FROZEN.md', 'parent_protocol_sha256')]:
        p = amendment / n
        assert sha(p) == prov[key]
        consumed[str(p)] = sha(p)
    af = read(amendment / 'P2J1_AMENDMENT_FREEZE.json')
    assert af['selected_alpha'] == 0.2 and af['selected_rho0'] == 0.0001 and (af['selected_a0'] == a0)
    write('P2L_REUSE_AUDIT.json', dict(status='EXACT_REUSED_ASSETS_VERIFIED', created_utc=utc(), package_sha256=package, consumed_sha256=consumed, model_file_sha256=modelhashes, model_dir=str(modeldir), model_commit=freeze['model_commit'], F0_path=binding['F0_file'], cross_hessian=binding['cross_hessian_path'], F0_shape=list(f0.shape), F0_array_sha256=array_sha(f0), u1_array_sha256=array_sha(u), v1_array_sha256=array_sha(v), s1=s, WD_norm=wd, rho0=freeze['rho0'], a0=a0, alpha=freeze['alpha'], old_assets_read_only=True, old_fitted_rates_opened=False, original_alpha_seal_status='FAIL_PRESERVED', operative_alpha_record='P2J1_ALPHA_GATE_REEVALUATED.json', counters=COUNTERS))
    preds = []
    for i, r in enumerate(freeze['r_eta']):
        T = np.eye(2) + freeze['alpha'] * np.array([[-1, 1], [r, -r * (1 - freeze['epsilon'])]])
        vals, vec = np.linalg.eig(T)
        assert np.max(np.abs(vals.imag)) == 0
        k = int(np.argmax(vals.real))
        e = vec[:, k].real
        e /= np.linalg.norm(e)
        if e[0] < 0:
            e = -e
        rho = float(vals[k])
        gamma = math.log(rho)
        N = 0
        while rho ** N < 1.25:
            N += 1
        endpoint = rho ** N
        ref = next((p for p in freeze['horizons'] if p['r_eta'] == r))
        assert N == ref['horizon_updates'] and abs(rho - ref['rho_plus']) < 1e-14 and (abs(gamma - ref['gamma_plus']) < 1e-14) and (abs(endpoint - ref['predicted_endpoint_amplitude_ratio']) < 1e-13)
        assert endpoint <= freeze['preflight_predicted_endpoint_max'] and abs(float(vals[1 - k])) < 1
        preds.append(dict(cell=i, epsilon=freeze['epsilon'], r_eta=r, s1=s, mu=s, lambda_=s * (1 - freeze['epsilon']), alpha=freeze['alpha'], eta_M=freeze['alpha'] / s, eta_U=r * freeze['alpha'] / s, rho_plus=rho, rho_minus=float(vals[1 - k]), gamma_plus=gamma, e_M=float(e[0]), e_U=float(e[1]), a0=a0, rho0=freeze['rho0'], N_cell=N, predicted_endpoint_amplitude_ratio=endpoint, T00=float(T[0, 0]), T01=float(T[0, 1]), T10=float(T[1, 0]), T11=float(T[1, 1]), primary_role='PRIMARY_MOBILITY_CONFIRMATION'))
    assert sum((2 * p['N_cell'] for p in preds)) == 78
    csvout('P2L_FROZEN_PREDICTIONS.csv', preds)
    sealed = dict(consumed, **package)
    for p in SOURCE.glob('*.py'):
        sealed[str(p)] = sha(p)
    for n in ['P2L_REUSE_AUDIT.json', 'P2L_FROZEN_PREDICTIONS.csv']:
        sealed[str(OUT / n)] = sha(OUT / n)
    write('P2L_PREFLIGHT_SEAL.json', dict(status='PASS', sealed_utc=utc(), sealed_sha256=sealed, source_sha256={str(p): sha(p) for p in SOURCE.glob('*.py')}, predictions_sha256=sha(OUT / 'P2L_FROZEN_PREDICTIONS.csv'), selected_alpha=0.2, selected_rho=0.0001, selected_a0=a0, primary_updates_before_seal=0, multistep_outcomes_opened_before_seal=0, total_primary_updates=78, counters=COUNTERS))
    print('M0/M1 PASS: exact assets; regenerated horizons ' + str([p['N_cell'] for p in preds]) + '; endpoint maxima ' + str(max((p['predicted_endpoint_amplitude_ratio'] for p in preds))), flush=True)
if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        write('P2L_STOP.json', dict(status='STOP_BEFORE_TRAINING', error_type=type(e).__name__, error=str(e), created_utc=utc(), counters=COUNTERS))
        raise
